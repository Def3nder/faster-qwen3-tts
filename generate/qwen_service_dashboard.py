#!/usr/bin/env python3
"""Status, Auftragsverlauf und Konsolen-Dashboard des Qwen-TTS-Dienstes; nur Standardbibliothek.

`ServiceState` sammelt, was der Dienst tut (HTTP-Anfragen, Pings vom WebArchiv, laufender
Auftrag, Verlauf). `Dashboard` zeichnet daraus ein stehendes Vollbild auf der Konsole
(ANSI/VT, auch Windows 10+), statt Zeilen nach oben zu scrollen. Die Detailausgaben stehen
im Log; der Verlauf überlebt Neustarts in einer kleinen JSON-Datei.
"""
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime

HISTORY_LIMIT = 200
EVENT_LIMIT = 300
PING_FLASH_SECONDS = 1.5
REQUEST_FLASH_SECONDS = 0.4
LEASE_WARNING_SECONDS = 30
CHUNK = re.compile(r"Chunk (\d+)/(\d+)")

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
GRAY = "\x1b[90m"
RED = "\x1b[91m"
GREEN = "\x1b[92m"
YELLOW = "\x1b[93m"
MAGENTA = "\x1b[95m"
CYAN = "\x1b[96m"
TITLE = "\x1b[1;97;44m"
FLASH_OK = "\x1b[1;30;102m"
FLASH_BAD = "\x1b[1;97;41m"
FLASH_REQUEST = "\x1b[1;30;106m"
IDLE_BADGE = "\x1b[90;100m"

OUTCOMES = {
    "ok": ("OK", GREEN),
    "failed": ("FEHLER", RED),
    "cancelled": ("ABBRUCH", YELLOW),
    "rejected": ("ABGELEHNT", MAGENTA),
}


WILDCARD_HOSTS = {"", "0.0.0.0", "::", "[::]"}


def local_addresses():
    """IPv4-Adressen dieses Rechners, die vom Netz erreichbar sind; die Hauptadresse zuerst.

    Hauptadresse = Schnittstelle der Standardroute (z. B. WLAN oder Ethernet). Ermittelt per UDP-
    „connect“ ins Dokumentationsnetz 192.0.2.0/24: Das wählt nur die Route, es wird nichts gesendet.
    Danach weitere Adressen (oft virtuelle Adapter von VMware, WSL, Hyper-V).
    """
    found = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            found.append(probe.getsockname()[0])
    except OSError:
        pass  # Kein Netz: dann nur die Adressliste des Rechnernamens.
    try:
        for entry in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(entry[4][0])
    except OSError:
        pass
    usable = []
    for address in found:
        if address and not address.startswith(("127.", "169.254.", "0.")) and address not in usable:
            usable.append(address)
    return usable


def listen_label(host, port, addresses):
    """„192.168.1.65:8765“ statt „0.0.0.0:8765“; ohne erkennbare Adresse bleibt die Bindung sichtbar."""
    if host in WILDCARD_HOSTS and addresses:
        return f"{addresses[0]}:{port}"
    return f"{host or '0.0.0.0'}:{port}"


def format_seconds(seconds):
    seconds = int(round(max(0, seconds)))
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {seconds:02d} s"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours} h {minutes:02d} min"
    days, hours = divmod(hours, 24)
    return f"{days} d {hours} h"


def format_bytes(size):
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size} Bytes"


def format_count(value):
    return f"{value:,}".replace(",", ".")


def clock(epoch, with_date=False):
    if not epoch:
        return "-"
    moment = datetime.fromtimestamp(epoch)
    if with_date:
        return moment.strftime("%d.%m. %H:%M")
    return moment.strftime("%H:%M:%S")


class ServiceState:
    """Thread-sicherer Zustand für das Dashboard; jede Änderung erhöht `version`."""

    def __init__(self, history_file=None, clock_source=time.time):
        self.lock = threading.RLock()
        self.time = clock_source
        self.started = self.time()
        self.version = 0
        self.info = {}
        self.requests = 0
        self.errors = 0
        self.last_request = None
        self.pings = 0
        self.last_ping = None
        self.unauthorized = 0
        self.last_unauthorized = None
        self.current = None
        self.voice = None
        self.history = deque(maxlen=HISTORY_LIMIT)
        self.events = deque(maxlen=EVENT_LIMIT)
        self.history_file = Path(history_file) if history_file else None
        self.history_error = None
        self._load_history()

    # ---- Persistenz -------------------------------------------------------------------------
    def _load_history(self):
        if not self.history_file or not self.history_file.is_file():
            return
        try:
            entries = json.loads(self.history_file.read_text(encoding="utf-8"))
            for entry in entries[-HISTORY_LIMIT:]:
                if isinstance(entry, dict) and entry.get("outcome") in OUTCOMES:
                    self.history.append(entry)
        except (OSError, ValueError, TypeError) as error:
            self.history_error = f"Verlauf nicht lesbar: {error}"

    def _save_history(self):
        if not self.history_file:
            return
        try:
            directory = self.history_file.resolve().parent
            descriptor, temporary = tempfile.mkstemp(prefix=f".{self.history_file.name}-", suffix=".tmp",
                                                     dir=directory)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
                    json.dump(list(self.history), output, ensure_ascii=False, indent=1)
                    output.write("\n")
                os.replace(temporary, self.history_file)
            finally:
                try:
                    Path(temporary).unlink()
                except FileNotFoundError:
                    pass
            self.history_error = None
        except OSError as error:
            self.history_error = f"Verlauf nicht gespeichert: {error}"

    # ---- Ereignisse -------------------------------------------------------------------------
    def _changed(self):
        self.version += 1

    def set_info(self, **values):
        with self.lock:
            self.info.update(values)
            self._changed()

    def add_event(self, level, moment, message):
        with self.lock:
            for index, line in enumerate(str(message).splitlines() or [""]):
                self.events.append((moment if index == 0 else None, level, line))
            self._changed()

    def request(self, method, path, status, ip, milliseconds, error=None, detail=None):
        with self.lock:
            now = self.time()
            self.requests += 1
            if status is None or status >= 400:
                self.errors += 1
            self.last_request = {"time": now, "method": method, "path": path, "status": status,
                                 "ip": ip, "ms": milliseconds, "error": error, "detail": detail}
            if path == "/v1/health":
                self.pings += 1
                self.last_ping = {"time": now, "ip": ip, "status": status, "detail": detail}
            if status == 401:
                self.unauthorized += 1
                self.last_unauthorized = {"time": now, "ip": ip, "path": path, "detail": detail}
            if self.current and path.startswith(f"/v1/jobs/{self.current['id']}") and status == 200:
                self.current["polled"] = now
            self._changed()

    def job_started(self, job_id, label, voice, characters, pid=None):
        with self.lock:
            now = self.time()
            self.current = {"id": job_id, "label": label, "voice": voice, "characters": characters,
                            "started": now, "polled": now, "pid": pid, "phase": "Worker startet",
                            "chunk": None, "chunks": None, "chunk_started": None, "message": None,
                            "worker_line": None}
            self._changed()

    def job_ready(self, job_id):
        with self.lock:
            if self.current and self.current["id"] == job_id:
                self.current["phase"] = "Generator lädt Modell"
                self._changed()

    def job_progress(self, job_id, message):
        with self.lock:
            if not self.current or self.current["id"] != job_id:
                return
            match = CHUNK.match(message)
            if match:
                self.current["chunk"], self.current["chunks"] = int(match[1]), int(match[2])
                self.current["chunk_started"] = self.time()
                if self.current.get("synthesis_started") is None:
                    self.current["synthesis_started"] = self.current["chunk_started"]
            self.current["phase"] = "Synthese läuft"
            self.current["message"] = message
            self._changed()

    def worker_line(self, line):
        with self.lock:
            if self.current:
                self.current["worker_line"] = line
                self._changed()

    def job_finished(self, job_id, outcome, reason=None, size=None, label=None):
        with self.lock:
            now = self.time()
            current = self.current if self.current and self.current["id"] == job_id else None
            entry = {
                "id": job_id, "kind": "article", "outcome": outcome,
                "label": (current or {}).get("label") or label or "?",
                "voice": (current or {}).get("voice"),
                "characters": (current or {}).get("characters"),
                "started": (current or {}).get("started", now), "ended": now,
                "chunks": (current or {}).get("chunks"), "chunk": (current or {}).get("chunk"),
                "size": size, "reason": reason, "fetched": False,
            }
            if current:
                self.current = None
            self.history.append(entry)
            self._save_history()
            self._changed()

    def job_rejected(self, job_id, label, reason, ip):
        with self.lock:
            now = self.time()
            self.history.append({"id": job_id, "kind": "article", "outcome": "rejected",
                                 "label": label or "?", "started": now, "ended": now,
                                 "reason": f"{reason} (von {ip})"})
            self._save_history()
            self._changed()

    def job_fetched(self, job_id):
        with self.lock:
            for entry in reversed(self.history):
                if entry.get("id") == job_id:
                    if not entry.get("fetched"):
                        entry["fetched"] = True
                        self._save_history()
                        self._changed()
                    return

    def voice_started(self, job_id, name):
        with self.lock:
            self.voice = {"id": job_id, "name": name, "started": self.time(), "stage": "wartet"}
            self._changed()

    def voice_stage(self, job_id, stage):
        with self.lock:
            if self.voice and self.voice["id"] == job_id:
                self.voice["stage"] = stage
                self._changed()

    def voice_finished(self, job_id, outcome, reason=None):
        with self.lock:
            now = self.time()
            voice = self.voice if self.voice and self.voice["id"] == job_id else {}
            self.history.append({"id": job_id, "kind": "voice", "outcome": outcome,
                                 "label": f"Stimme {voice.get('name', '?')}",
                                 "started": voice.get("started", now), "ended": now, "reason": reason})
            if voice:
                self.voice = None
            self._save_history()
            self._changed()

    def snapshot(self):
        with self.lock:
            return {
                "now": self.time(), "started": self.started, "info": dict(self.info),
                "requests": self.requests, "errors": self.errors,
                "last_request": dict(self.last_request) if self.last_request else None,
                "pings": self.pings, "last_ping": dict(self.last_ping) if self.last_ping else None,
                "unauthorized": self.unauthorized,
                "last_unauthorized": dict(self.last_unauthorized) if self.last_unauthorized else None,
                "current": dict(self.current) if self.current else None,
                "voice": dict(self.voice) if self.voice else None,
                "history": [dict(entry) for entry in self.history],
                "events": list(self.events), "history_error": self.history_error,
            }


# ---- Darstellung ------------------------------------------------------------------------------

def compose(segments, width):
    """Segmente (Text, Stil) auf `width` sichtbare Zeichen kürzen; Stile zählen nicht mit."""
    output, remaining = [], width
    for text, style in segments:
        if remaining <= 0:
            break
        text = str(text).replace("\t", " ").replace("\r", " ").replace("\n", " ")
        if len(text) > remaining:
            text = text[:max(0, remaining - 1)] + "…"
        remaining -= len(text)
        output.append(f"{style}{text}{RESET}" if style else text)
    return "".join(output)


def rule(title, width, style=CYAN):
    title = f"── {title} "
    return [(title, style + BOLD), ("─" * max(0, width - len(title)), style)]


def progress_bar(done, total, width):
    filled = 0 if not total else round(width * min(1.0, done / total))
    return [("█" * filled, GREEN), ("░" * (width - filled), GRAY)]


def history_line(entry, width):
    outcome, color = OUTCOMES.get(entry.get("outcome"), ("?", ""))
    segments = [(f" {clock(entry.get('ended'), with_date=True)}  ", GRAY), (f"{outcome:<9} ", color + BOLD)]
    segments.append((f"\"{entry.get('label', '?')}\"" if entry.get("kind") != "voice" else entry.get("label", "?"), ""))
    details = []
    if entry.get("outcome") != "rejected":
        details.append(format_seconds((entry.get("ended") or 0) - (entry.get("started") or 0)))
    if entry.get("size"):
        details.append(format_bytes(entry["size"]))
    if entry.get("chunks") and entry.get("outcome") == "ok":
        details.append(f"{entry['chunks']} Chunks")
    elif entry.get("chunks"):
        details.append(f"bis Chunk {entry.get('chunk') or '?'}/{entry['chunks']}")
    if entry.get("voice"):
        details.append(f"Stimme {entry['voice']}")
    if entry.get("outcome") == "ok" and entry.get("kind") != "voice":
        details.append("abgeholt" if entry.get("fetched") else "nicht abgeholt")
    if details:
        segments.append(("  " + " · ".join(details), GRAY))
    if entry.get("reason"):
        segments.append(("  " + str(entry["reason"]), color))
    return compose(segments, width)


def render(snapshot, width, height):
    """Ein vollständiges Bild als Liste von Zeilen (mit ANSI-Stilen), höchstens `height` Zeilen."""
    width = max(40, width)
    now = snapshot["now"]
    info = snapshot["info"]
    top = []

    def add(*segments):
        top.append(compose(segments, width))

    level = info.get("debug", 0)
    add((" Qwen-TTS-Dienst ", TITLE), ("  ● läuft ", GREEN + BOLD),
        (f" {listen_label(info.get('host', '?'), info.get('port', '?'), info.get('addresses') or [])}", BOLD),
        (" (alle Schnittstellen)" if info.get("host") in WILDCARD_HOSTS else "", GRAY),
        (f"   {info.get('protocol', '')}", GRAY),
        (f"   seit {format_seconds(now - snapshot['started'])} (Start {clock(snapshot['started'], True)})", ""),
        (f"   PID {info.get('pid', '?')} · Debug {level}", GRAY))
    log_file = info.get("log_file")
    add((f" Log: {log_file} (Level {max(level, 1)})" if log_file else " Log: aus", GRAY))
    if snapshot.get("history_error"):
        add((" " + snapshot["history_error"], RED))
    top.append(compose(rule("Verbindung", width), width))

    ping = snapshot["last_ping"]
    ping_flash = ping and now - ping["time"] < PING_FLASH_SECONDS
    ping_ok = ping and ping["status"] == 200
    ping_style = (FLASH_OK if ping_ok else FLASH_BAD) if ping_flash else IDLE_BADGE
    if ping:
        ping_text = (f"  Health-Check {clock(ping['time'])} von {ping['ip']} (vor {format_seconds(now - ping['time'])})"
                     f" -> {ping['status']}" + (f" [{ping['detail']}]" if ping.get("detail") else "")
                     + f" · {format_count(snapshot['pings'])} Pings")
    else:
        ping_text = "  Noch kein Health-Check vom WebArchiv"
    add((" WebArchiv ", BOLD), ("  PING  ", ping_style), (ping_text, "" if ping_ok or not ping else RED))

    last = snapshot["last_request"]
    request_flash = last and now - last["time"] < REQUEST_FLASH_SECONDS
    request_bad = last and (last["status"] is None or last["status"] >= 400)
    request_style = (FLASH_BAD if request_bad else FLASH_REQUEST) if request_flash else IDLE_BADGE
    totals = f"  {format_count(snapshot['requests'])} Anfragen · {format_count(snapshot['errors'])} mit Fehler"
    add((" HTTP      ", BOLD), (" ANFRAGE ", request_style), (totals, ""))
    if last:
        text = (f"{last['method']} {last['path']} von {last['ip']} -> {last['status']} "
                f"({last['ms']:.0f} ms, vor {format_seconds(now - last['time'])})")
        if last.get("error"):
            text += f": {last['error']}"
        if last.get("detail"):
            text += f" [{last['detail']}]"
        add(("            letzte: ", GRAY), (text, RED if request_bad else ""))
    if snapshot["unauthorized"]:
        bad = snapshot["last_unauthorized"]
        add(("            ", ""), (f"{snapshot['unauthorized']}× nicht autorisiert, zuletzt {clock(bad['time'])} "
                                   f"von {bad['ip']}: {bad.get('detail') or ''}", RED))

    current, voice = snapshot["current"], snapshot["voice"]
    if current:
        add((" GPU       ", BOLD), (" BELEGT ", FLASH_BAD), (f"  Artikel \"{current['label']}\"", ""))
    elif voice:
        add((" GPU       ", BOLD), (" BELEGT ", FLASH_BAD), (f"  Stimmerzeugung \"{voice['name']}\"", ""))
    else:
        add((" GPU       ", BOLD), ("  FREI  ", FLASH_OK), ("  wartet auf Aufträge", GRAY))

    top.append(compose(rule("Aktueller Auftrag", width), width))
    if current:
        voice_text = f"Stimme {current['voice']}" if current.get("voice") else "Standardstimme"
        add((f" \"{current['label']}\"", BOLD),
            (f"  [{current['id'][:8]}] · {voice_text} · {format_count(current['characters'] or 0)} Zeichen", GRAY))
        elapsed = now - current["started"]
        poll_age = now - current["polled"]
        add((f" Gestartet {clock(current['started'])} · läuft seit {format_seconds(elapsed)} · ", ""),
            (f"letzte Statusabfrage vor {format_seconds(poll_age)}",
             RED if poll_age > LEASE_WARNING_SECONDS else GRAY),
            (f" (Abbruch nach {info.get('lease', 90)} s ohne Abfrage)", GRAY))
        if current.get("chunks"):
            done = current["chunk"] - 1
            eta = ""
            if done > 0:
                # Ab dem ersten Chunk messen: das Laden des Modells zählt nicht zur Restzeit.
                synthesis = now - (current.get("synthesis_started") or current["started"])
                remaining = synthesis / done * (current["chunks"] - done)
                eta = f" · noch ca. {format_seconds(remaining)}"
            bar_width = max(10, min(40, width - 60))
            add((" ", ""), *progress_bar(done, current["chunks"], bar_width),
                (f" Chunk {current['chunk']}/{current['chunks']}{eta}", BOLD),
                (f"  {current['message']}", GRAY))
        else:
            add((f" {current['phase']} …", YELLOW))
        worker = f" Worker PID {current['pid']}" if current.get("pid") else " Worker"
        add((worker + " · letzte Ausgabe: ", GRAY), (current.get("worker_line") or "-", ""))
    elif voice:
        add((f" Stimme \"{voice['name']}\"", BOLD), (f"  [{voice['id'][:8]}] · {voice['stage']} · seit "
                                                    f"{format_seconds(now - voice['started'])}", GRAY))
    else:
        add((" Kein Auftrag – der Dienst wartet auf das WebArchiv.", GRAY))

    history = snapshot["history"]
    today = datetime.fromtimestamp(now).date()
    todays = [entry for entry in history if entry.get("ended")
              and datetime.fromtimestamp(entry["ended"]).date() == today]
    counts = {key: sum(1 for entry in todays if entry.get("outcome") == key) for key in OUTCOMES}
    produced = sum(entry.get("size") or 0 for entry in todays if entry.get("outcome") == "ok")
    summary = (f"Verlauf · heute {counts['ok']} OK, {counts['failed']} Fehler, {counts['cancelled']} Abbruch, "
               f"{counts['rejected']} abgelehnt, {format_bytes(produced)} MP3 · gesamt {len(history)}")

    footer = compose([(" Strg+C beendet den Dienst", GRAY),
                      (f"   {datetime.fromtimestamp(now).strftime('%d.%m.%Y %H:%M:%S')}", GRAY)], width)
    free = height - len(top) - 3  # zwei Trennlinien + Fußzeile
    events = snapshot["events"]
    event_rows = min(len(events), max(3, free // 3))
    history_rows = max(1, free - event_rows)
    if len(history) < history_rows:
        event_rows += history_rows - max(1, len(history))
        history_rows = max(1, len(history))
    event_rows = max(0, min(event_rows, free - history_rows))

    lines = list(top)
    lines.append(compose(rule(summary, width), width))
    if history:
        lines.extend(history_line(entry, width) for entry in reversed(history[-history_rows:]))
    else:
        lines.append(compose([(" Noch keine Aufträge.", GRAY)], width))
    lines.append(compose(rule("Ereignisse (Details im Log)", width), width))
    for moment, event_level, text in events[-event_rows:] if event_rows else []:
        prefix = clock(moment) if moment else "        "
        style = RED if ("FEHLER" in text or "ABGELEHNT" in text) else (GRAY if event_level >= 2 else "")
        lines.append(compose([(f" {prefix} ", GRAY), (text, style)], width))
    while len(lines) < height - 1:
        lines.append("")
    lines = lines[:max(1, height - 1)]
    lines.append(footer)
    return lines


def enable_virtual_terminal(stream=None):
    """ANSI-Steuerzeichen auf der Windows-Konsole einschalten; False, wenn das nicht geht."""
    stream = stream or sys.stdout
    try:
        if not stream.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))  # VIRTUAL_TERMINAL_PROCESSING
    except (AttributeError, OSError):
        return False


class Dashboard:
    """Zeichnet `render()` im alternativen Bildschirmpuffer neu, ohne zu scrollen."""

    def __init__(self, state, stream=None, interval=0.25):
        self.state = state
        self.stream = stream or sys.stdout
        self.interval = interval
        self.stopped = threading.Event()
        self.thread = None
        self.last_frame = None
        self.last_size = None

    def start(self):
        self.stream.write("\x1b[?1049h\x1b[?25l\x1b[2J")
        self.stream.flush()
        self.thread = threading.Thread(target=self.run, name="dashboard", daemon=True)
        self.thread.start()

    def draw(self):
        size = shutil.get_terminal_size((120, 40))
        lines = render(self.state.snapshot(), size.columns - 1, size.lines)
        frame = "\x1b[H" + "\n".join(line + "\x1b[K" for line in lines) + "\x1b[J"
        if size != self.last_size:
            frame = "\x1b[2J" + frame
            self.last_size = size
        if frame != self.last_frame:
            self.stream.write(frame)
            self.stream.flush()
            self.last_frame = frame

    def run(self):
        while not self.stopped.is_set():
            try:
                self.draw()
            except Exception as error:  # Das Dashboard darf den Dienst nie stoppen.
                self.state.add_event(0, time.time(), f"Dashboard-Fehler: {type(error).__name__}: {error}")
            self.stopped.wait(self.interval)

    def stop(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=2)
        try:
            self.stream.write("\x1b[?25h\x1b[?1049l")
            self.stream.flush()
        except (OSError, ValueError):
            pass
