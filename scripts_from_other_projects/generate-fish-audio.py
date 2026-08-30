import argparse
import html
import os
import re
import sys
from pathlib import Path

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

API_KEY = os.environ["FISH_API_KEY"]
REFRENCE_ID = "6a978a8dec384998b84da952d037f31e"


MD_SUFFIXES = {".md", ".markdown", ".mdown", ".mkd"}

# Platzhalter für Zeilen ohne sprechbaren Inhalt (Leerzeilen, Satzzeichen-Zeilen,
# Trennlinien). "" = Zeile ersatzlos entfernen, keine Leerzeile bleibt zurück.
# PAUSE_MARKER = "[]"
PAUSE_MARKER = "{pause}"

# Ersetzungen für Zeichen/Abkürzungen, die das TTS-System falsch ausspricht.
# Längere Varianten zuerst. Nach Bedarf kürzen oder ergänzen.
REPLACEMENTS = {
    "z. B.": "zum Beispiel",
    "z.\u00a0B.": "zum Beispiel",   # mit geschütztem Leerzeichen
    "z.B.": "zum Beispiel",
    "d. h.": "das heißt",
    "d.\u00a0h.": "das heißt",
    "d.h.": "das heißt",
    "u. a.": "unter anderem",
    "u.a.": "unter anderem",
    "ggf.": "gegebenenfalls",
    "bzw.": "beziehungsweise",
    "§": "Paragraf ",
    "€": " Euro",
    "&": " und ",
    "Joe Turan": "Joe Turahn",
}


def _inline(s: str) -> str:
    """Inline-Markdown innerhalb einer Zeile entfernen."""
    s = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", s)        # Bilder -> Alt-Text
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)         # Links  -> Linktext
    s = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", s)        # Referenz-Links
    s = re.sub(r"<https?://[^>]+>", "", s)                 # Autolinks raus
    s = re.sub(r"`+([^`]+)`+", r"\1", s)                   # Inline-Code
    s = re.sub(r"(\*\*\*|___)(\S.*?\S|\S)\1", r"\2", s)    # fett+kursiv
    s = re.sub(r"(\*\*|__)(\S.*?\S|\S)\1", r"\2", s)       # fett
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", s)   # kursiv
    s = re.sub(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])", r"\1", s)     # kursiv
    s = re.sub(r"~~(.+?)~~", r"\1", s)                     # durchgestrichen
    s = re.sub(r"<[^>]+>", "", s)                          # HTML-Tags
    return html.unescape(s)


def strip_markdown(text: str, bullet: str = "") -> str:
    """Markdown-Syntax entfernen. Nummerierte Listen bleiben erhalten.

    bullet: Ersatz für Aufzählungszeichen am Zeilenanfang ("" = entfernen).
    """
    bullet = bullet.replace("\\", r"\\")  # als re.sub-Replacement absichern

    text = text.replace("\r\n", "\n")
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)               # YAML-Frontmatter
    text = re.sub(r"^(```|~~~).*?^\1[^\n]*$", "", text, flags=re.S | re.M)  # Codeblöcke
    text = re.sub(r"^([^\n]+)\n=+[ \t]*$", r"\1", text, flags=re.M)         # Setext-H1
    text = re.sub(r"^([^\n]+)\n-{3,}[ \t]*$", r"\1", text, flags=re.M)      # Setext-H2

    out = []

    def add_pause():
        """Platzhalter setzen, ohne Doppelungen. Bei leerem Marker: nichts."""
        if PAUSE_MARKER and (not out or out[-1] != PAUSE_MARKER):
            out.append(PAUSE_MARKER)

    for line in text.split("\n"):
        # Leerzeilen, reine Satzzeichen, Trennlinien, Tabellen-Trennzeilen:
        # kein sprechbarer Inhalt.
        if not re.search(r"\w", line):
            add_pause()
            continue

        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)         # Überschriften
        line = re.sub(r"\s+#+\s*$", "", line)                 # closing hashes
        line = re.sub(r"^\s*(?:>\s?)+", "", line)             # Zitate
        line = re.sub(r"^(\s*)\|(.*)\|\s*$", r"\1\2", line)   # Tabellenränder
        if "|" in line:
            line = line.replace(" | ", ", ")

        m = re.match(r"^\s*(\d+)[.)]\s+(.*)$", line)          # nummerierte Liste: behalten
        if m:
            line = f"{m.group(1)}. {m.group(2)}"
        else:
            line = re.sub(r"^\s*[-*+]\s+\[[ xX]\]\s*", bullet, line)  # Task-Liste
            line = re.sub(r"^\s*[-*+]\s+", bullet, line)              # Aufzählung

        line = re.sub(r"[ \t]{2,}", " ", _inline(line))       # Auffüll-Leerzeichen
        line = re.sub(r"\s+([,.;:])", r"\1", line)
        line = line.strip()

        if not re.search(r"\w", line):        # nach dem Strippen nichts übrig
            add_pause()
            continue

        out.append(line)

    if PAUSE_MARKER:
        while out and out[0] == PAUSE_MARKER:     # führende Marker weg
            out.pop(0)
        while out and out[-1] == PAUSE_MARKER:    # abschließende Marker weg
            out.pop()

    return "\n".join(out)


def normalize_numbers(s: str) -> str:
    """Deutschen Tausenderpunkt entfernen: 1.499,95 -> 1499,95"""
    return re.sub(
        r"(?<!\d)\d{1,3}(?:\.\d{3})+(?!\d)",
        lambda m: m.group(0).replace(".", ""),
        s,
    )


def apply_replacements(s: str) -> str:
    """Zeichen und Abkürzungen ausschreiben."""
    for a, b in REPLACEMENTS.items():
        s = s.replace(a, b)
    return re.sub(r"[ \t]{2,}", " ", s)


def unique_path(path: Path) -> Path:
    """Freien Dateinamen finden: name.mp3, name_2.mp3, name_3.mp3, ..."""
    if not path.exists():
        return path
    n = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


parser = argparse.ArgumentParser(description="Text- oder Markdown-Datei via Fish Audio TTS vertonen")
parser.add_argument("--file", required=True, help="Pfad zur Eingabedatei")
parser.add_argument("--out", help="Ziel-Audiodatei (Default: Eingabename mit .mp3)")
parser.add_argument(
    "--bullet",
    default="",
    help="Ersatz für Aufzählungszeichen am Zeilenanfang. "
         "Default: leer (Marker wird entfernt). Beispiel: --bullet=\"Punkt: \"",
)
parser.add_argument("--no-strip", action="store_true", help="Markdown nicht bereinigen")
parser.add_argument("--no-fix", action="store_true", help="Zahlen/Abkürzungen nicht ersetzen")
parser.add_argument("--dry-run", action="store_true", help="nur den bereinigten Text ausgeben")
args = parser.parse_args()

src = Path(args.file)
text = src.read_text(encoding="utf-8-sig")

if not args.no_strip and src.suffix.lower() in MD_SUFFIXES:
    text = strip_markdown(text, bullet=args.bullet)

if not args.no_fix:
    text = normalize_numbers(text)
    text = apply_replacements(text)

if not text.strip():
    raise SystemExit("Fehler: Nach der Bereinigung ist kein Text übrig.")

if args.dry_run:
    print(text)
    raise SystemExit(0)

out_path = Path(args.out) if args.out else unique_path(src.with_suffix(".mp3"))

body = {
    "text": text,
    "reference_id": REFRENCE_ID,
    "format": "mp3",
}

with httpx.Client(timeout=120) as client:
    res = client.post(
        "https://api.fish.audio/v1/tts",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "model": "s2.1-pro-free",
        },
        json=body,
    )
res.raise_for_status()
out_path.write_bytes(res.content)
print(f"{len(res.content)} Bytes geschrieben nach {out_path}")