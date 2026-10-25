#!/usr/bin/env python3
"""HTTP-Brücke für generate_mp3_with_embedding.py; nur Python-Standardbibliothek."""
import argparse
import base64
import binascii
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import wave
from array import array
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

PROTOCOL = "webarchiv-qwen-v5"
MAX_TEXT = 1_000_000
LEASE = 90
RETENTION = 3600
PROGRESS_HISTORY = 100
MAX_VOICE_BYTES = 10 * 1024 * 1024
VOICE_MIN_SECONDS = 3.0
VOICE_MAX_SECONDS = 60.0
UUID_PATTERN = r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
CHUNK_PROGRESS = re.compile(r"^Chunk \d+/\d+: \d+/\d+ characters \(boundary: [^)]+\)$")
VOICE_ID = re.compile(r"^[\w .-]{1,80}$", re.UNICODE)
WINDOWS_DEVICE_NAME = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)", re.IGNORECASE)
VOICE_MIME_EXTENSIONS = {
    "audio/webm": ".webm", "audio/ogg": ".ogg", "audio/wav": ".wav",
    "audio/x-wav": ".wav", "audio/mpeg": ".mp3", "audio/mp4": ".m4a",
    "audio/aac": ".aac", "application/octet-stream": ".bin",
}
SERVICE_CONFIG_NAME = "qwen_http_service_config.json"
SERVER_SETTING_DEFAULTS = {
    "target_chunk_chars": 340,
    "min_chunk_chars": 220,
    "max_chunk_chars": 520,
    "text_preroll_enabled": True,
    "text_preroll_sentence": "Der Tag kommt zu einem Ende.\n\n.",
    "append_chunk_end_padding": True,
    "chunk_end_padding_text": "\n\n.",
    "text_preroll_min_pause_ms": 120,
    "text_preroll_lead_in_ms": 30,
    "silence_threshold_db": -50.0,
}


def _read_config(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Die Qwen-Konfiguration muss ein JSON-Objekt sein")
    return data


def _atomic_json(path, data):
    path = Path(path).resolve()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(data, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass


def validate_server_settings(data):
    if not isinstance(data, dict) or set(data) != set(SERVER_SETTING_DEFAULTS):
        raise ValueError("Unvollständige Server-Einstellungen")
    values = dict(data)
    for key in ("min_chunk_chars", "target_chunk_chars", "max_chunk_chars"):
        if type(values[key]) is not int or not 50 <= values[key] <= 2000:
            raise ValueError("Chunk-Längen müssen ganze Zahlen zwischen 50 und 2000 sein")
    if not values["min_chunk_chars"] <= values["target_chunk_chars"] <= values["max_chunk_chars"]:
        raise ValueError("Chunk-Längen müssen Min ≤ Ziel ≤ Max erfüllen")
    for key in ("text_preroll_enabled", "append_chunk_end_padding"):
        if type(values[key]) is not bool:
            raise ValueError("Schalter müssen boolesche Werte sein")
    for key in ("text_preroll_sentence", "chunk_end_padding_text"):
        if not isinstance(values[key], str) or len(values[key]) > 2000:
            raise ValueError("Preroll- und Padding-Text dürfen höchstens 2000 Zeichen lang sein")
    if values["text_preroll_enabled"] and not values["text_preroll_sentence"].strip():
        raise ValueError("Für den Text-Preroll wird ein Satz benötigt")
    if values["append_chunk_end_padding"] and not values["chunk_end_padding_text"]:
        raise ValueError("Für das Chunk-Ende wird ein Padding-Text benötigt")
    for key in ("text_preroll_min_pause_ms", "text_preroll_lead_in_ms"):
        if type(values[key]) is not int or not 0 <= values[key] <= 5000:
            raise ValueError("Pausenwerte müssen ganze Millisekunden zwischen 0 und 5000 sein")
    threshold = values["silence_threshold_db"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not -100 <= threshold <= 0:
        raise ValueError("Der Stille-Schwellenwert muss zwischen -100 und 0 dB liegen")
    values["silence_threshold_db"] = float(threshold)
    return values


class ServerSettings:
    def __init__(self, config):
        self.config = Path(config).resolve()
        self.lock = threading.RLock()

    def get(self):
        with self.lock:
            config = _read_config(self.config)
            values = {key: config.get(key, default) for key, default in SERVER_SETTING_DEFAULTS.items()}
            return validate_server_settings(values)

    def update(self, data):
        values = validate_server_settings(data)
        with self.lock:
            config = _read_config(self.config)
            config.update(values)
            _atomic_json(self.config, config)
        return values


class GpuCoordinator:
    """Serialisiert Artikelsynthese und Stimmerzeugung auf derselben GPU."""

    def __init__(self):
        self.lock = threading.RLock()
        self.owner = None

    def acquire(self, owner):
        with self.lock:
            if self.owner is not None and self.owner != owner:
                raise ValueError("GPU belegt")
            self.owner = owner

    def release(self, owner):
        with self.lock:
            if self.owner == owner:
                self.owner = None

    @property
    def busy(self):
        with self.lock:
            return self.owner is not None


def safe_voice_id(value):
    value = unicodedata.normalize("NFKC", str(value or "")).strip()
    if (not VOICE_ID.fullmatch(value) or value in {".", ".."}
            or "/" in value or "\\" in value or value.endswith((" ", "."))
            or WINDOWS_DEVICE_NAME.match(value)):
        raise ValueError("Ungültiger Stimmenname")
    return value


def voice_name(value):
    value = unicodedata.normalize("NFKC", str(value or "")).strip()
    if value.casefold().endswith("-1.7b.pt"):
        value = value[:-len("-1.7B.pt")]
    value = re.sub(r"[^\w .-]+", "-", value, flags=re.UNICODE)
    value = re.sub(r"[\s_-]+", "-", value).strip("-. ")
    return safe_voice_id(value)


class VoiceCatalog:
    def __init__(self, directory, model="Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
        self.directory = Path(directory).resolve()
        self.model = model
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, voice_id, must_exist=True):
        voice_id = safe_voice_id(voice_id)
        candidate = self.directory / f"{voice_id}-1.7B.pt"
        if candidate.parent.resolve() != self.directory:
            raise ValueError("Ungültiger Stimmenpfad")
        if must_exist and (not candidate.is_file() or candidate.is_symlink()):
            raise KeyError(voice_id)
        return candidate

    def list(self):
        result = []
        for candidate in sorted(self.directory.glob("*-1.7B.pt"), key=lambda p: p.name.casefold()):
            if not candidate.is_file() or candidate.is_symlink() or candidate.parent.resolve() != self.directory:
                continue
            stat = candidate.stat()
            voice_id = candidate.name[:-len("-1.7B.pt")]
            try:
                safe_voice_id(voice_id)
            except ValueError:
                continue
            digest = hashlib.sha256()
            with candidate.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            result.append({
                "id": voice_id, "name": voice_id, "model": self.model,
                "createdAt": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": digest.hexdigest(),
            })
        return result

    def resolve(self, voice_id):
        return self.path(voice_id, must_exist=True)

    def rename(self, current_id, requested_name):
        source = self.resolve(current_id)
        target_id = voice_name(requested_name)
        target = self.path(target_id, must_exist=False)
        if target.exists():
            raise FileExistsError(target_id)
        source.rename(target)
        return {"id": target_id, "name": target_id, "model": self.model}

    def trash(self, voice_id):
        source = self.resolve(voice_id)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        destination_dir = self.directory / "trash" / stamp
        suffix = 1
        while destination_dir.exists():
            suffix += 1
            destination_dir = self.directory / "trash" / f"{stamp}-{suffix}"
        destination_dir.mkdir(parents=True)
        source.rename(destination_dir / source.name)
        return {"id": voice_id, "trashed": True}


class VoiceJobs:
    def __init__(self, catalog, extract_script, coordinator, python=sys.executable,
                 ffmpeg="ffmpeg", device="cuda:0"):
        self.catalog = catalog
        self.extract_script = Path(extract_script).resolve()
        self.coordinator = coordinator
        self.python = python
        self.ffmpeg = ffmpeg
        self.device = device
        self.lock = threading.RLock()
        self.jobs = {}
        self.closed = False

    @staticmethod
    def _decode_audio(value):
        if not isinstance(value, str) or len(value) > 4 * MAX_VOICE_BYTES // 3 + 16:
            raise ValueError("Ungültige Aufnahme")
        try:
            audio = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("Ungültige Aufnahme") from error
        if not 0 < len(audio) <= MAX_VOICE_BYTES:
            raise ValueError("Ungültige Aufnahmegröße")
        return audio

    def start(self, job_id, requested_name, mime_type, audio_base64):
        target_id = voice_name(requested_name)
        mime_type = str(mime_type or "").split(";", 1)[0].strip().lower()
        extension = VOICE_MIME_EXTENSIONS.get(mime_type)
        if extension is None:
            raise ValueError("Nicht unterstütztes Audioformat")
        audio = self._decode_audio(audio_base64)
        digest = hashlib.sha256(target_id.encode() + b"\0" + audio).hexdigest()
        owner = f"voice:{job_id}"
        with self.lock:
            existing = self.jobs.get(job_id)
            if existing:
                if existing.get("digest") != digest:
                    raise ValueError("ID bereits belegt")
                return self.status(job_id)
            if self.closed or len(self.jobs) >= 100:
                raise ValueError("Stimmenauftragsspeicher belegt")
            if self.catalog.path(target_id, must_exist=False).exists():
                raise FileExistsError(target_id)
            self.coordinator.acquire(owner)
            directory = None
            try:
                directory = Path(tempfile.mkdtemp(prefix=".webarchiv-voice-", dir=self.catalog.directory))
                source = directory / f"recording{extension}"
                source.write_bytes(audio)
                job = {
                    "id": job_id, "name": target_id, "status": "running", "stage": "queued",
                    "error": None, "digest": digest, "directory": directory, "source": source,
                    "process": None, "cancelled": False, "touched": time.monotonic(), "owner": owner,
                }
                self.jobs[job_id] = job
                threading.Thread(target=self._run, args=(job,), daemon=True).start()
                return self.status(job_id)
            except Exception:
                if directory:
                    shutil.rmtree(directory, ignore_errors=True)
                self.coordinator.release(owner)
                raise

    def _process(self, job, args, timeout):
        if job["cancelled"]:
            raise InterruptedError()
        process = subprocess.Popen(
            args, cwd=str(self.extract_script.parent.parent),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        with self.lock:
            job["process"] = process
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            Jobs.stop_process(process)
            raise ValueError("Zeitlimit der Stimmerzeugung überschritten")
        finally:
            with self.lock:
                job["process"] = None
        if job["cancelled"]:
            raise InterruptedError()
        if code != 0:
            raise ValueError("Stimmerzeugung fehlgeschlagen")

    @staticmethod
    def _validate_wav(path):
        with wave.open(str(path), "rb") as source:
            if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getframerate() != 24000:
                raise ValueError("Ungültiges Referenz-WAV")
            frames = source.getnframes()
            duration = frames / source.getframerate()
            if not VOICE_MIN_SECONDS <= duration <= VOICE_MAX_SECONDS:
                raise ValueError("Die Aufnahme muss zwischen 3 und 60 Sekunden lang sein")
            samples = array("h", source.readframes(frames))
            if sys.byteorder == "big":
                samples.byteswap()
            rms = math.sqrt(sum(sample * sample for sample in samples) / max(1, len(samples)))
            if rms < 100:
                raise ValueError("Die Aufnahme enthält keine ausreichend laute Sprache")
        return duration

    def _run(self, job):
        try:
            with self.lock:
                job["stage"] = "converting"
            wav = job["directory"] / "reference.wav"
            self._process(job, [
                self.ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(job["source"]),
                "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(wav),
            ], 120)
            duration = self._validate_wav(wav)
            with self.lock:
                job["stage"] = "extracting"
                job["duration"] = duration
            temporary_embedding = job["directory"] / f"{job['name']}-1.7B.pt"
            self._process(job, [
                self.python, str(self.extract_script), "--ref_audio", str(wav),
                "--model_path", self.catalog.model, "--device", self.device,
                "--output", str(temporary_embedding),
            ], 900)
            if (not temporary_embedding.is_file()
                    or not 1024 <= temporary_embedding.stat().st_size <= 1024 * 1024):
                raise ValueError("Das erzeugte Stimmen-Embedding ist ungültig")
            target = self.catalog.path(job["name"], must_exist=False)
            if target.exists():
                raise FileExistsError(job["name"])
            os.link(temporary_embedding, target)
            with self.lock:
                job["status"] = "succeeded"
                job["stage"] = "done"
                job["voice"] = {"id": job["name"], "name": job["name"], "model": self.catalog.model}
        except InterruptedError:
            with self.lock:
                job["status"] = "cancelled"
                job["stage"] = "cancelled"
        except FileExistsError:
            with self.lock:
                job["status"] = "failed"
                job["error"] = "Eine Stimme mit diesem Namen existiert bereits."
        except Exception as error:
            with self.lock:
                job["status"] = "failed"
                job["error"] = str(error) or "Stimmerzeugung fehlgeschlagen."
        finally:
            shutil.rmtree(job["directory"], ignore_errors=True)
            self.coordinator.release(job["owner"])
            with self.lock:
                job["process"] = None
                job["touched"] = time.monotonic()

    def status(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            job["touched"] = time.monotonic()
            return {key: job.get(key) for key in ("id", "name", "status", "stage", "error", "duration", "voice") if job.get(key) is not None}

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] != "running":
                return self.status(job_id)
            job["cancelled"] = True
            process = job.get("process")
        if process:
            Jobs.stop_process(process)
        return self.status(job_id)

    def sweep(self):
        with self.lock:
            now = time.monotonic()
            for job_id, job in list(self.jobs.items()):
                if job["status"] != "running" and now - job["touched"] > RETENTION:
                    del self.jobs[job_id]

    def close(self):
        with self.lock:
            self.closed = True
            running = [job["id"] for job in self.jobs.values() if job["status"] == "running"]
        for job_id in running:
            self.cancel(job_id)


class Jobs:
    def __init__(self, script, config, python=sys.executable, worker=None, coordinator=None, catalog=None):
        self.script = str(Path(script).resolve())
        self.config = str(Path(config).resolve())
        self.python = python
        self.worker = str(Path(worker or Path(__file__).with_name("qwen_chunk_worker.py")).resolve())
        self.coordinator = coordinator or GpuCoordinator()
        self.catalog = catalog
        self.lock = threading.RLock()
        self.jobs = {}
        self.sessions = {}
        self.closed = False

    def start(self, job_id, text, session_id=None, chunk_index=1, total_chunks=1, voice_id=None):
        session_id = session_id or job_id
        if not text.strip() or len(text.encode('utf-8')) > MAX_TEXT:
            raise ValueError("Ungültige Markdown-Länge")
        if type(chunk_index) is not int or type(total_chunks) is not int or chunk_index != 1 or total_chunks != 1:
            raise ValueError("Es wird ausschließlich ein vollständiger Artikel akzeptiert")
        if voice_id is not None and self.catalog is None:
            raise ValueError("Stimmenauswahl wird nicht unterstützt")
        if voice_id is not None:
            safe_voice_id(voice_id)
        speaker = None
        digest = hashlib.sha256(json.dumps([text, session_id, chunk_index, total_chunks, voice_id]).encode()).hexdigest()
        with self.lock:
            existing = self.jobs.get(job_id)
            if existing:
                if existing.get("digest") != digest:
                    raise ValueError("ID bereits belegt oder abgebrochen")
                return self.status(job_id)
            if self.closed or any(j["status"] == "running" for j in self.jobs.values()):
                raise ValueError("Dienst belegt")
            if len(self.jobs) >= 1000 or len(self.sessions) >= 1000:
                raise ValueError("Auftragsspeicher belegt")
            session = self.sessions.get(session_id)
            if session is None:
                if chunk_index != 1 or any(s["status"] == "active" for s in self.sessions.values()):
                    raise ValueError("Sitzung fehlt oder GPU belegt")
                owner = f"article:{session_id}"
                self.coordinator.acquire(owner)
                try:
                    if voice_id is not None:
                        speaker = self.catalog.resolve(voice_id)
                    process = subprocess.Popen(
                        [self.python, "-u", self.worker, "--script", self.script, "--config", self.config],
                        cwd=str(Path(self.script).parent), stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                        text=True, encoding="utf-8", bufsize=1,
                        start_new_session=os.name != "nt",
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                except Exception:
                    self.coordinator.release(owner)
                    raise
                session = {"id": session_id, "process": process, "status": "active",
                           "total": total_chunks, "next": 1, "ready": False,
                           "owner": owner, "touched": time.monotonic()}
                self.sessions[session_id] = session
                threading.Thread(target=self.monitor, args=(session,), daemon=True).start()
            if session["status"] != "active" or session["next"] != chunk_index or session["total"] != total_chunks:
                raise ValueError("Ungültige oder bereits beendete Sitzung/Chunkreihenfolge")
            process = session["process"]
            directory = Path(tempfile.mkdtemp(prefix="webarchiv-qwen-"))
            try:
                source, output = directory / "input.md", directory / "output.mp3"
                source.write_bytes(text.encode("utf-8"))
                job = {"id": job_id, "status": "running", "digest": digest,
                       "session": session_id, "index": chunk_index,
                       "process": process, "directory": directory, "output": output,
                       "progress": [], "progress_sequence": 0,
                       "touched": time.monotonic()}
                self.jobs[job_id] = job
                # Dateipfade entstehen ausschließlich hier; HTTP-Clients liefern nur Text/IDs.
                command = {"id": job_id, "input": str(source), "output": str(output)}
                if speaker is not None:
                    command["speaker"] = str(speaker)
                process.stdin.write(json.dumps(command) + "\n")
                process.stdin.flush()
                session["next"] += 1
            except Exception:
                self.end_session(session_id)
                shutil.rmtree(directory, ignore_errors=True)
                raise
            return self.status(job_id)

    def monitor(self, session):
        process = session["process"]
        try:
            for line in process.stdout:
                event = json.loads(line)
                with self.lock:
                    if session["status"] != "active":
                        continue
                    if event.get("type") == "ready" and not session["ready"]:
                        session["ready"] = True
                    elif event.get("type") == "progress":
                        job = self.jobs[event.get("id")]
                        message = event.get("message")
                        if (job["session"] != session["id"] or job["status"] != "running"
                                or not session["ready"] or not isinstance(message, str)
                                or not CHUNK_PROGRESS.fullmatch(message)):
                            raise ValueError("Ungültiger Worker-Fortschritt")
                        job["progress_sequence"] += 1
                        job["progress"].append({
                            "sequence": job["progress_sequence"], "message": message,
                        })
                        del job["progress"][:-PROGRESS_HISTORY]
                        job["touched"] = time.monotonic()
                    elif event.get("type") == "result":
                        job = self.jobs[event["id"]]
                        if job["session"] != session["id"] or job["status"] != "running" or not session["ready"]:
                            raise ValueError("Inkonsistentes Worker-Ergebnis")
                        output = job["output"]
                        if not output.is_file() or output.stat().st_size == 0:
                            raise ValueError("Leeres Worker-Ergebnis")
                        job["status"] = "succeeded"
                        job["touched"] = time.monotonic()
                    else:
                        raise ValueError("Ungültige Worker-Nachricht")
        except Exception:
            self.stop_process(process)
        finally:
            process.wait()
            process.stdout.close()
            self.coordinator.release(session.get("owner"))
            with self.lock:
                if session["status"] == "active":
                    session["status"] = "finished" if process.returncode == 0 else "failed"
                for job in self.jobs.values():
                    if job.get("session") == session["id"] and job["status"] == "running":
                        job["status"] = "failed"

    @staticmethod
    def stop_process(process):
        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=3, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=3)
        if process.stdin:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass

    def end_session(self, session_id):
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                if len(self.sessions) >= 1000:
                    raise ValueError("Sitzungsspeicher belegt")
                self.sessions[session_id] = {"id": session_id, "status": "closed", "touched": time.monotonic()}
                return
            if session.get("process"):
                self.stop_process(session["process"])
            session["status"] = "closed"
            session["touched"] = time.monotonic()
            for job in self.jobs.values():
                if job.get("session") == session_id:
                    if job["status"] == "running":
                        job["status"] = "cancelled"
                    if job.get("directory"):
                        shutil.rmtree(job["directory"], ignore_errors=True)

    def status(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            job["touched"] = time.monotonic()
            session = self.sessions.get(job.get("session"))
            if session:
                session["touched"] = time.monotonic()
            return {"id": job_id, "status": job["status"],
                    "progress": list(job.get("progress", []))}

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                if len(self.jobs) >= 1000:
                    raise ValueError("Auftragsspeicher belegt")
                # Verhindert verspäteten PUT nach verloren gegangener Startantwort.
                self.jobs[job_id] = {"id": job_id, "status": "cancelled", "touched": time.monotonic()}
                return
            if job["status"] == "cancelled":
                return
            session = self.sessions.get(job.get("session"))
            if session:
                session["touched"] = time.monotonic()
                if job["status"] != "succeeded" or job["index"] == session["total"]:
                    self.end_session(session["id"])
            job["status"] = "cancelled"
            job["touched"] = time.monotonic()
            if job.get("directory"):
                shutil.rmtree(job["directory"], ignore_errors=True)

    def sweep(self):
        with self.lock:
            now = time.monotonic()
            for session_id, session in list(self.sessions.items()):
                age = now - session["touched"]
                if session["status"] == "active" and age > LEASE:
                    self.end_session(session_id)
                elif session["status"] != "active" and age > RETENTION:
                    del self.sessions[session_id]
            for job_id, job in list(self.jobs.items()):
                age = now - job["touched"]
                if job["status"] == "running" and age > LEASE:
                    self.cancel(job_id)
                elif job["status"] != "running" and age > RETENTION:
                    if job.get("directory"):
                        shutil.rmtree(job["directory"], ignore_errors=True)
                    del self.jobs[job_id]

    def close(self):
        with self.lock:
            self.closed = True
            for session_id in list(self.sessions):
                self.end_session(session_id)


def handler(jobs, token, voices=None, settings=None):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, *_args):
            pass  # Keine Token, Texte oder Dateipfade in HTTP-Logs.

        def json(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def read_json(self, maximum):
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ValueError("Ungültige Request-Länge") from error
            if self.headers.get("Transfer-Encoding") or size < 1 or size > maximum:
                raise OverflowError("Ungültige Request-Länge")
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise ValueError("Unvollständiger Request")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Ungültiges JSON-Objekt")
            return data

        def dispatch(self):
            if not hmac.compare_digest(self.headers.get("Authorization", "").encode(), f"Bearer {token}".encode()):
                return self.json(401, {"error": "Nicht autorisiert"})
            request_path = urlsplit(self.path).path
            if self.command == "GET" and request_path == "/v1/health":
                return self.json(200, {"protocol": PROTOCOL, "lease_seconds": LEASE,
                                       "full_markdown": True, "audio_format": "mp3",
                                       "chunk_progress": True, "max_markdown_bytes": MAX_TEXT,
                                       "voice_catalog": voices is not None,
                                       "voice_creation": voices is not None,
                                       "server_settings": settings is not None,
                                       "voice_model": voices.catalog.model if voices else None,
                                       "max_voice_upload_bytes": MAX_VOICE_BYTES if voices else None})
            if settings and request_path == "/v1/settings":
                try:
                    if self.command == "GET":
                        return self.json(200, {"settings": settings.get(), "busy": jobs.coordinator.busy})
                    if self.command == "PUT":
                        data = self.read_json(16384)
                        return self.json(200, {"settings": settings.update(data), "busy": jobs.coordinator.busy})
                    return self.json(405, {"error": "Methode nicht erlaubt"})
                except OverflowError:
                    return self.json(413, {"error": "Request ist zu groß."})
                except (ValueError, UnicodeError) as error:
                    return self.json(400, {"error": str(error)})
            if voices and self.command == "GET" and request_path == "/v1/voices":
                return self.json(200, {"voices": voices.catalog.list(), "model": voices.catalog.model,
                                       "busy": jobs.coordinator.busy})
            voice_match = re.fullmatch(r"/v1/voices/([^/]+)", request_path)
            if voices and voice_match:
                voice_id = unquote(voice_match[1])
                try:
                    if jobs.coordinator.busy:
                        return self.json(409, {"error": "Die GPU ist durch einen Auftrag belegt."})
                    if self.command == "PATCH":
                        data = self.read_json(4096)
                        if set(data) != {"name"}:
                            raise ValueError("Ungültige Umbenennung")
                        return self.json(200, voices.catalog.rename(voice_id, data["name"]))
                    if self.command == "DELETE":
                        return self.json(200, voices.catalog.trash(voice_id))
                    return self.json(405, {"error": "Methode nicht erlaubt"})
                except KeyError:
                    return self.json(404, {"error": "Stimme nicht gefunden."})
                except FileExistsError:
                    return self.json(409, {"error": "Eine Stimme mit diesem Namen existiert bereits."})
                except (ValueError, UnicodeError) as error:
                    return self.json(400, {"error": str(error)})
                except OverflowError:
                    return self.json(413, {"error": "Request ist zu groß."})
            voice_job_match = re.fullmatch(r"/v1/voice-jobs/(" + UUID_PATTERN + r")", request_path)
            if voices and voice_job_match:
                job_id = voice_job_match[1]
                try:
                    if self.command == "PUT":
                        data = self.read_json(4 * MAX_VOICE_BYTES // 3 + 65536)
                        if set(data) != {"name", "mimeType", "audioBase64"}:
                            raise ValueError("Ungültiger Stimmenauftrag")
                        return self.json(202, voices.start(job_id, data["name"], data["mimeType"], data["audioBase64"]))
                    if self.command == "GET":
                        return self.json(200, voices.status(job_id))
                    if self.command == "DELETE":
                        return self.json(200, voices.cancel(job_id))
                    return self.json(405, {"error": "Methode nicht erlaubt"})
                except KeyError:
                    return self.json(404, {"error": "Stimmenauftrag unbekannt."})
                except FileExistsError:
                    return self.json(409, {"error": "Eine Stimme mit diesem Namen existiert bereits."})
                except OverflowError:
                    return self.json(413, {"error": "Aufnahme ist zu groß."})
                except (ValueError, UnicodeError) as error:
                    return self.json(409, {"error": str(error)})
            session_match = re.fullmatch(r"/v1/sessions/(" + UUID_PATTERN + r")", request_path)
            if session_match and self.command == "DELETE":
                try:
                    jobs.end_session(session_match[1])
                    return self.json(200, {"id": session_match[1], "status": "closed"})
                except Exception:
                    return self.json(500, {"error": "Sitzung konnte nicht beendet werden"})
            match = re.fullmatch(r"/v1/jobs/(" + UUID_PATTERN + r")(/audio)?", request_path)
            if not match:
                return self.json(404, {"error": "Unbekannter Endpunkt"})
            job_id, audio = match.groups()
            try:
                if self.command == "PUT" and not audio:
                    data = self.read_json(6 * MAX_TEXT + 400)
                    if (not set(data).issubset({"markdown", "voiceId"}) or "markdown" not in data
                            or not isinstance(data["markdown"], str) or not data["markdown"].strip()
                            or len(data["markdown"].encode('utf-8')) > MAX_TEXT
                            or ("voiceId" in data and not isinstance(data["voiceId"], str))):
                        return self.json(400, {"error": "Ungültiges Markdown"})
                    try:
                        started = jobs.start(job_id, data["markdown"], voice_id=data.get("voiceId"))
                    except KeyError:
                        return self.json(409, {"error": "Die gewählte Stimme ist nicht verfügbar."})
                    return self.json(202, started)
                if self.command == "DELETE" and not audio:
                    jobs.cancel(job_id)
                    return self.json(200, {"id": job_id, "status": "cancelled"})
                if self.command == "GET" and not audio:
                    return self.json(200, jobs.status(job_id))
                if self.command == "GET" and audio:
                    with jobs.lock:
                        if jobs.status(job_id)["status"] != "succeeded":
                            return self.json(409, {"error": "Audio nicht fertig"})
                        with jobs.jobs[job_id]["output"].open("rb") as source:
                            self.send_response(200)
                            self.send_header("Content-Type", "audio/mpeg")
                            self.send_header("Content-Length", str(os.fstat(source.fileno()).st_size))
                            self.send_header("Cache-Control", "no-store")
                            self.end_headers()
                            shutil.copyfileobj(source, self.wfile)
                    return
                self.json(405, {"error": "Methode nicht erlaubt"})
            except KeyError:
                self.json(404, {"error": "Auftrag unbekannt"})
            except OverflowError:
                self.json(413, {"error": "Ungültige Request-Länge"})
            except (ValueError, UnicodeError):
                self.json(409, {"error": "Ungültiger oder konkurrierender Auftrag"})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            except Exception:
                self.json(500, {"error": "Qwen-Dienstfehler"})

        do_GET = dispatch
        do_PUT = dispatch
        do_DELETE = dispatch
        do_PATCH = dispatch
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--voices-dir", type=Path)
    parser.add_argument("--extract-script", type=Path)
    parser.add_argument("--voice-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--voice-device", default="cuda:0")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    token = os.environ.get("QWEN_TTS_TOKEN", "").strip()
    if args.config.name != SERVICE_CONFIG_NAME:
        parser.error(f"--config muss auf die vorhandene {SERVICE_CONFIG_NAME} zeigen; config.json wird nicht verwendet.")
    if bool(args.voices_dir) != bool(args.extract_script):
        parser.error("--voices-dir und --extract-script müssen gemeinsam angegeben werden.")
    if (len(token) < 32 or not args.script.is_file() or not args.config.is_file()
            or not Path(__file__).with_name("qwen_chunk_worker.py").is_file()
            or (args.extract_script and not args.extract_script.is_file())):
        parser.error("Generator, Konfiguration und benachbarter qwen_chunk_worker.py müssen existieren; QWEN_TTS_TOKEN benötigt mindestens 32 Zeichen.")
    coordinator = GpuCoordinator()
    voice_jobs = None
    catalog = None
    if args.voices_dir:
        catalog = VoiceCatalog(args.voices_dir, args.voice_model)
        voice_jobs = VoiceJobs(
            catalog, args.extract_script, coordinator, ffmpeg=args.ffmpeg,
            device=args.voice_device,
        )
    settings = ServerSettings(args.config)
    jobs = Jobs(args.script, args.config, coordinator=coordinator, catalog=catalog)
    server = ThreadingHTTPServer((args.host, args.port), handler(jobs, token, voice_jobs, settings))
    print(json.dumps({"protocol": PROTOCOL, "port": server.server_port}), flush=True)
    stopped = threading.Event()

    def maintenance():
        while not stopped.wait(1):
            try:
                jobs.sweep()
                if voice_jobs:
                    voice_jobs.sweep()
            except Exception:
                # Abbruchfehler nicht als Erfolg melden; beim nächsten Tick erneut versuchen.
                print("Qwen-Aufräumen fehlgeschlagen; erneuter Versuch folgt.", file=sys.stderr)

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    threading.Thread(target=maintenance, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stopped.set()
        server.server_close()
        jobs.close()
        if voice_jobs:
            voice_jobs.close()


if __name__ == "__main__":
    main()
