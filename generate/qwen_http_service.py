#!/usr/bin/env python3
"""HTTP-Brücke für generate_mp3_with_embedding.py; nur Python-Standardbibliothek."""
import argparse
import hashlib
import hmac
import json
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL = "webarchiv-qwen-v4"
MAX_TEXT = 1_000_000
LEASE = 90
RETENTION = 3600
UUID_PATTERN = r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"


class Jobs:
    def __init__(self, script, config, python=sys.executable, worker=None):
        self.script = str(Path(script).resolve())
        self.config = str(Path(config).resolve())
        self.python = python
        self.worker = str(Path(worker or Path(__file__).with_name("qwen_chunk_worker.py")).resolve())
        self.lock = threading.RLock()
        self.jobs = {}
        self.sessions = {}
        self.closed = False

    def start(self, job_id, text, session_id=None, chunk_index=1, total_chunks=1):
        session_id = session_id or job_id
        if not text.strip() or len(text.encode('utf-8')) > MAX_TEXT:
            raise ValueError("Ungültige Markdown-Länge")
        if type(chunk_index) is not int or type(total_chunks) is not int or chunk_index != 1 or total_chunks != 1:
            raise ValueError("Es wird ausschließlich ein vollständiger Artikel akzeptiert")
        digest = hashlib.sha256(json.dumps([text, session_id, chunk_index, total_chunks]).encode()).hexdigest()
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
                process = subprocess.Popen(
                    [self.python, "-u", self.worker, "--script", self.script, "--config", self.config],
                    cwd=str(Path(self.script).parent), stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    text=True, encoding="utf-8", bufsize=1,
                    start_new_session=os.name != "nt",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                session = {"id": session_id, "process": process, "status": "active",
                           "total": total_chunks, "next": 1, "ready": False, "touched": time.monotonic()}
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
                       "touched": time.monotonic()}
                self.jobs[job_id] = job
                # Dateipfade entstehen ausschließlich hier; HTTP-Clients liefern nur Text/IDs.
                process.stdin.write(json.dumps({"id": job_id, "input": str(source), "output": str(output)}) + "\n")
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
            return {"id": job_id, "status": job["status"]}

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


def handler(jobs, token):
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

        def dispatch(self):
            if not hmac.compare_digest(self.headers.get("Authorization", "").encode(), f"Bearer {token}".encode()):
                return self.json(401, {"error": "Nicht autorisiert"})
            if self.command == "GET" and self.path == "/v1/health":
                return self.json(200, {"protocol": PROTOCOL, "lease_seconds": LEASE,
                                       "full_markdown": True, "audio_format": "mp3", "max_markdown_bytes": MAX_TEXT})
            session_match = re.fullmatch(r"/v1/sessions/(" + UUID_PATTERN + r")", self.path)
            if session_match and self.command == "DELETE":
                try:
                    jobs.end_session(session_match[1])
                    return self.json(200, {"id": session_match[1], "status": "closed"})
                except Exception:
                    return self.json(500, {"error": "Sitzung konnte nicht beendet werden"})
            match = re.fullmatch(r"/v1/jobs/(" + UUID_PATTERN + r")(/audio)?", self.path)
            if not match:
                return self.json(404, {"error": "Unbekannter Endpunkt"})
            job_id, audio = match.groups()
            try:
                if self.command == "PUT" and not audio:
                    size = int(self.headers.get("Content-Length", "0"))
                    if self.headers.get("Transfer-Encoding") or size < 1 or size > 6 * MAX_TEXT + 400:
                        return self.json(413, {"error": "Ungültige Request-Länge"})
                    raw = self.rfile.read(size)
                    if len(raw) != size:
                        return self.json(400, {"error": "Unvollständiger Request"})
                    data = json.loads(raw)
                    if not isinstance(data, dict) or set(data) != {"markdown"} or not isinstance(data["markdown"], str) or not data["markdown"].strip() or len(data["markdown"].encode('utf-8')) > MAX_TEXT:
                        return self.json(400, {"error": "Ungültiges Markdown"})
                    return self.json(202, jobs.start(job_id, data["markdown"]))
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
            except (ValueError, UnicodeError):
                self.json(409, {"error": "Ungültiger oder konkurrierender Auftrag"})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            except Exception:
                self.json(500, {"error": "Qwen-Dienstfehler"})

        do_GET = dispatch
        do_PUT = dispatch
        do_DELETE = dispatch
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    token = os.environ.get("QWEN_TTS_TOKEN", "").strip()
    if len(token) < 32 or not args.script.is_file() or not args.config.is_file() or not Path(__file__).with_name("qwen_chunk_worker.py").is_file():
        parser.error("Generator, Konfiguration und benachbarter qwen_chunk_worker.py müssen existieren; QWEN_TTS_TOKEN benötigt mindestens 32 Zeichen.")
    jobs = Jobs(args.script, args.config)
    server = ThreadingHTTPServer((args.host, args.port), handler(jobs, token))
    print(json.dumps({"protocol": PROTOCOL, "port": server.server_port}), flush=True)
    stopped = threading.Event()

    def maintenance():
        while not stopped.wait(1):
            try:
                jobs.sweep()
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


if __name__ == "__main__":
    main()
