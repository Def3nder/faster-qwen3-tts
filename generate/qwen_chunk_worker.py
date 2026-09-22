#!/usr/bin/env python3
"""Vollständiges Markdown über das Originalscript in eine fertige MP3 umwandeln."""
import argparse
import runpy
from pathlib import Path
import sys
import json
from contextlib import redirect_stdout


def generate_article(script, config, source, output):
    previous_argv = sys.argv
    previous_path = sys.path[:]
    try:
        # Genau der reguläre CLI-Aufruf. Keine Overrides für Markdown, Chunking,
        # Seed, Warmup, Preroll, Sprache oder Audioexport.
        sys.argv = [str(script), "--config", str(config), "--input", str(source), "--output", str(output)]
        sys.path.insert(0, str(script.parent))
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise RuntimeError(f"Qwen-Originalscript beendet: Exit {error.code}") from error
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    # stdout ist ausschließlich das interne NDJSON-Protokoll. Modellmeldungen
    # dürfen keine Ergebnisnachricht vortäuschen oder den Reader blockieren.
    print(json.dumps({"type": "ready"}), flush=True)
    command = json.loads(sys.stdin.readline())
    source = Path(command["input"])
    if not 0 < source.stat().st_size <= 1_000_000:
        raise ValueError("Markdown muss 1 bis 1.000.000 UTF-8-Bytes enthalten")
    with redirect_stdout(sys.stderr):
        generate_article(args.script.resolve(), args.config.resolve(), source, Path(command["output"]))
    print(json.dumps({"type": "result", "id": command["id"]}), flush=True)


if __name__ == "__main__":
    main()
