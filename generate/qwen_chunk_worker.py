#!/usr/bin/env python3
"""Vollständiges Markdown über das Originalscript in eine fertige MP3 umwandeln."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import json
from contextlib import redirect_stdout


CHUNK_PROGRESS = re.compile(
    r"^Chunk \d+/\d+: \d+/\d+ characters \(boundary: [^)]+\)$"
)


class ChunkProgressRelay:
    """Verwirft Modellmeldungen und reicht nur vollständige Chunk-Starts als NDJSON weiter."""
    def __init__(self, protocol, job_id):
        self.protocol = protocol
        self.job_id = job_id
        self.buffer = ""

    def write(self, value):
        self.buffer += value
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._emit(line.rstrip("\r"))
        return len(value)

    def flush(self):
        self.protocol.flush()

    def finish(self):
        if self.buffer:
            self._emit(self.buffer.rstrip("\r"))
            self.buffer = ""
        self.protocol.flush()

    def isatty(self):
        return False

    def _emit(self, line):
        if CHUNK_PROGRESS.fullmatch(line):
            self.protocol.write(json.dumps({
                "type": "progress", "id": self.job_id, "message": line,
            }) + "\n")
            self.protocol.flush()


def generate_article(script, config, source, output, speaker=None):
    # Das Originalskript bewusst als echten CLI-Prozess starten. Insbesondere
    # Modell-, CUDA- und Audiozustand dürfen nicht mit dem Protokoll-Worker
    # geteilt werden; so entspricht die Synthese exakt einem lokalen Aufruf.
    command = [
        sys.executable, "-u", str(script),
        "--config", str(config), "--input", str(source), "--output", str(output),
    ]
    if speaker is not None:
        command.extend(["--speaker", str(speaker)])
    process = subprocess.Popen(
        command, cwd=str(script.parent), stdout=subprocess.PIPE,
        stderr=None, text=True, encoding="utf-8", errors="replace", bufsize=1,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Qwen-Originalscript beendet: Exit {return_code}")
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    # stdout ist ausschließlich das interne NDJSON-Protokoll. Modellmeldungen
    # dürfen keine Ergebnisnachricht vortäuschen oder den Reader blockieren.
    protocol = sys.stdout
    print(json.dumps({"type": "ready"}), file=protocol, flush=True)
    command = json.loads(sys.stdin.readline())
    source = Path(command["input"])
    if not 0 < source.stat().st_size <= 1_000_000:
        raise ValueError("Markdown muss 1 bis 1.000.000 UTF-8-Bytes enthalten")
    progress = ChunkProgressRelay(protocol, command["id"])
    try:
        with redirect_stdout(progress):
            speaker = command.get("speaker")
            generate_article(
                args.script.resolve(), args.config.resolve(), source,
                Path(command["output"]), Path(speaker) if speaker else None,
            )
    finally:
        progress.finish()
    print(json.dumps({"type": "result", "id": command["id"]}), file=protocol, flush=True)


if __name__ == "__main__":
    main()
