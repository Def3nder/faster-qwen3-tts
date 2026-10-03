"""Core logic for the Markdown audiobook batch UI.

This module deliberately has no Qt dependency.  It owns the text that is shown
in the preview and later handed to ``generate_mp3_with_embedding.py`` so the UI
cannot silently generate something different from what it displayed.
"""

from __future__ import annotations

import csv
import html
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ATX_HEADING_RE = re.compile(
    r"^(?P<indent> {0,3})(?P<marks>#{1,6})[ \t]+(?P<title>.*?)[ \t]*#*[ \t]*$"
)
FOOTNOTE_START_RE = re.compile(r"^\[\^(?P<id>[^\]]+)\]:[ \t]*(?P<text>.*)$")
FOOTNOTE_REFERENCE_RE = re.compile(r"\[\^(?P<id>[^\]]+)\]")
TOC_LINK_RE = re.compile(
    r"(?m)^\s*[-*+]\s+\[(?P<title>[^\]]+)\]\(#(?P<id>[^)]+)\)\s*$"
)
HTML_ANCHOR_RE = re.compile(
    r"<a\b[^>]*\b(?:id|name)\s*=\s*(?P<quote>[\"'])(?P<id>.*?)(?P=quote)[^>]*>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class Heading:
    level: int
    title: str
    start: int
    end: int
    line: int


@dataclass(frozen=True)
class TocAnchor:
    identifier: str
    title: str
    start: int
    line: int


@dataclass
class Segment:
    key: str
    title: str
    start: int
    end: int
    source_text: str
    kind: str = "chapter"
    included_by_default: bool = True
    speech_text: str = ""
    output_name: str = ""
    status: str = "Bereit"


@dataclass(frozen=True)
class Replacement:
    source: str
    target: str
    note: str = ""


@dataclass
class SpeechOptions:
    speak_footnotes: bool = True
    pronunciation_entries: Sequence[Replacement] = field(default_factory=tuple)
    project_replacements: Sequence[Replacement] = field(default_factory=tuple)
    hyphenate_number_words: bool = True


def read_utf8_text(path: Path) -> str:
    """Read a UTF-8 Markdown file, accepting an optional BOM."""
    return path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")


def _without_inline_markdown(text: str) -> str:
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[`*_~]", "", text)
    return html.unescape(text).strip()


def parse_headings(text: str) -> list[Heading]:
    """Return ATX headings outside fenced code blocks with exact offsets."""
    headings: list[Heading] = []
    offset = 0
    fence: tuple[str, int] | None = None
    for line_number, line_with_end in enumerate(text.splitlines(keepends=True), start=1):
        line = line_with_end.rstrip("\r\n")
        fence_match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = (marker[0], len(marker))
            elif marker[0] == fence[0] and len(marker) >= fence[1]:
                fence = None
            offset += len(line_with_end)
            continue
        if fence is None:
            match = ATX_HEADING_RE.match(line)
            if match:
                title = _without_inline_markdown(match.group("title")) or "Unbenannt"
                headings.append(
                    Heading(
                        level=len(match.group("marks")),
                        title=title,
                        start=offset,
                        end=offset + len(line_with_end),
                        line=line_number,
                    )
                )
        offset += len(line_with_end)
    return headings


def parse_toc_anchors(text: str) -> list[TocAnchor]:
    """Resolve bulleted in-document TOC links to their HTML anchor positions."""
    positions: dict[str, tuple[int, int]] = {}
    for match in HTML_ANCHOR_RE.finditer(text):
        identifier = html.unescape(match.group("id")).strip()
        if not identifier or identifier in positions:
            continue
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_number = text.count("\n", 0, line_start) + 1
        positions[identifier] = (line_start, line_number)

    chapters: list[TocAnchor] = []
    seen: set[str] = set()
    for match in TOC_LINK_RE.finditer(text):
        identifier = html.unescape(match.group("id")).strip()
        if identifier in seen or identifier not in positions:
            continue
        seen.add(identifier)
        start, line = positions[identifier]
        chapters.append(
            TocAnchor(
                identifier=identifier,
                title=_without_inline_markdown(match.group("title")) or identifier,
                start=start,
                line=line,
            )
        )
    return sorted(chapters, key=lambda item: item.start)


def _body_without_first_heading(text: str) -> str:
    lines = text.splitlines()
    if lines and ATX_HEADING_RE.match(lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _body_without_headings(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not ATX_HEADING_RE.match(line)).strip()


def _segment_key(kind: str, start: int, part: int = 1) -> str:
    return f"{kind}:{start}:{part}"


def build_segments(
    text: str,
    split_level: int = 3,
    manual_splits: Iterable[int] = (),
) -> list[Segment]:
    """Split Markdown at one heading level and optional character offsets.

    A heading at a higher level closes the current chapter.  Text outside the
    requested heading level remains visible as a separately selectable intro or
    interlude instead of disappearing silently.
    """
    if not 1 <= split_level <= 6:
        raise ValueError("split_level must be between 1 and 6")
    headings = parse_headings(text)
    target_headings = [h for h in headings if h.level == split_level]
    base: list[Segment] = []

    if not target_headings:
        if text.strip():
            base.append(
                Segment(
                    key=_segment_key("intro", 0),
                    title="Einleitung",
                    start=0,
                    end=len(text),
                    source_text=text,
                    kind="intro",
                    included_by_default=False,
                )
            )
    else:
        first_target = target_headings[0]
        if text[: first_target.start].strip():
            base.append(
                Segment(
                    key=_segment_key("intro", 0),
                    title="Einleitung",
                    start=0,
                    end=first_target.start,
                    source_text=text[: first_target.start],
                    kind="intro",
                    included_by_default=False,
                )
            )
        for index, heading in enumerate(target_headings):
            next_target_start = (
                target_headings[index + 1].start if index + 1 < len(target_headings) else len(text)
            )
            closing_heading = next(
                (
                    candidate
                    for candidate in headings
                    if heading.start < candidate.start < next_target_start
                    and candidate.level <= split_level
                ),
                None,
            )
            chapter_end = closing_heading.start if closing_heading else next_target_start
            base.append(
                Segment(
                    key=_segment_key("chapter", heading.start),
                    title=heading.title,
                    start=heading.start,
                    end=chapter_end,
                    source_text=text[heading.start:chapter_end],
                )
            )
            if chapter_end < next_target_start:
                raw = text[chapter_end:next_target_start]
                if not _body_without_headings(raw):
                    continue
                interlude_heading = next(
                    (candidate for candidate in headings if candidate.start == chapter_end),
                    None,
                )
                title = interlude_heading.title if interlude_heading else "Zwischentext"
                base.append(
                    Segment(
                        key=_segment_key("interlude", chapter_end),
                        title=f"Zwischentext – {title}",
                        start=chapter_end,
                        end=next_target_start,
                        source_text=raw,
                        kind="interlude",
                        included_by_default=False,
                    )
                )

    return _apply_manual_splits(base, text, manual_splits)


def _apply_manual_splits(
    base: Sequence[Segment], text: str, manual_splits: Iterable[int]
) -> list[Segment]:
    split_offsets = sorted({int(value) for value in manual_splits if 0 < int(value) < len(text)})
    result: list[Segment] = []
    for segment in base:
        cuts = [value for value in split_offsets if segment.start < value < segment.end]
        if not cuts:
            result.append(segment)
            continue
        positions = [segment.start, *cuts, segment.end]
        for part, (start, end) in enumerate(zip(positions, positions[1:]), start=1):
            piece = text[start:end]
            if not piece.strip():
                continue
            result.append(
                Segment(
                    key=_segment_key(segment.kind, segment.start, part),
                    title=segment.title if part == 1 else f"{segment.title} – Teil {part}",
                    start=start,
                    end=end,
                    source_text=piece,
                    kind=segment.kind,
                    included_by_default=segment.included_by_default,
                )
            )
    return result


def build_toc_anchor_segments(
    text: str,
    manual_splits: Iterable[int] = (),
) -> list[Segment]:
    """Split a document using TOC links and their matching HTML anchors."""
    chapters = parse_toc_anchors(text)
    base: list[Segment] = []
    if not chapters:
        if text.strip():
            base.append(
                Segment(
                    key=_segment_key("intro", 0),
                    title="Einleitung",
                    start=0,
                    end=len(text),
                    source_text=text,
                    kind="intro",
                    included_by_default=False,
                )
            )
        return _apply_manual_splits(base, text, manual_splits)

    if text[: chapters[0].start].strip():
        base.append(
            Segment(
                key=_segment_key("intro", 0),
                title="Einleitung",
                start=0,
                end=chapters[0].start,
                source_text=text[: chapters[0].start],
                kind="intro",
                included_by_default=False,
            )
        )
    for index, chapter in enumerate(chapters):
        end = chapters[index + 1].start if index + 1 < len(chapters) else len(text)
        base.append(
            Segment(
                key=_segment_key("anchor", chapter.start),
                title=chapter.title,
                start=chapter.start,
                end=end,
                source_text=text[chapter.start:end],
                kind="chapter",
            )
        )
    return _apply_manual_splits(base, text, manual_splits)


def paragraph_start(text: str, position: int) -> int:
    """Snap an arbitrary cursor position to the current paragraph start."""
    position = min(max(position, 0), len(text))
    before = text[:position]
    matches = list(re.finditer(r"\n[ \t]*\n", before))
    return matches[-1].end() if matches else 0


def set_segment_range(segment: Segment, document: str, start: int, end: int) -> None:
    """Replace a segment's suggested range with an exact user selection."""
    if not 0 <= start < end <= len(document):
        raise ValueError("Der markierte Bereich liegt außerhalb des Dokuments.")
    selected = document[start:end]
    if not selected.strip():
        raise ValueError("Der markierte Bereich enthält keinen vorlesbaren Text.")
    segment.start = start
    segment.end = end
    segment.source_text = selected


def safe_filename(title: str, fallback: str = "Abschnitt") -> str:
    title = unicodedata.normalize("NFC", title)
    title = title.translate(
        str.maketrans(
            {
                "Ä": "Ae",
                "Ö": "Oe",
                "Ü": "Ue",
                "ä": "ae",
                "ö": "oe",
                "ü": "ue",
                "ẞ": "SS",
                "ß": "ss",
            }
        )
    )
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", title)
    title = re.sub(r"[\s.]+", "_", title).strip("_ ")
    return (title[:120].rstrip("_ ") or fallback)


def normalize_output_title(value: str, fallback: str = "Abschnitt") -> str:
    """Extract and sanitize the editable title part of an MP3 filename."""
    value = value.strip()
    if re.search(r"(?i)\.mp3$", value):
        value = re.sub(r"(?i)\.mp3$", "", value)
        value = re.sub(r"^\d{3,}_", "", value)
    return safe_filename(value, fallback)


def assign_output_names(
    segments: Sequence[Segment],
    included: Sequence[bool],
    title_overrides: Mapping[str, str] | None = None,
) -> None:
    count = sum(bool(value) for value in included)
    width = max(3, len(str(max(1, count))))
    number = 0
    overrides = title_overrides or {}
    for segment, is_included in zip(segments, included):
        if not is_included:
            segment.output_name = "—"
            continue
        number += 1
        title = overrides.get(segment.key, segment.title)
        segment.output_name = f"{number:0{width}d}_{safe_filename(title)}.mp3"


def extract_footnotes(text: str) -> tuple[str, dict[str, str]]:
    """Remove Markdown footnote definitions and return them by identifier."""
    lines = text.splitlines()
    output: list[str] = []
    footnotes: dict[str, str] = {}
    index = 0
    while index < len(lines):
        match = FOOTNOTE_START_RE.match(lines[index])
        if not match:
            output.append(lines[index])
            index += 1
            continue
        identifier = match.group("id")
        parts = [match.group("text").strip()]
        index += 1
        while index < len(lines):
            continuation = lines[index]
            if re.match(r"^(?: {2,}|\t)\S", continuation):
                parts.append(continuation.strip())
                index += 1
                continue
            if not continuation.strip() and index + 1 < len(lines) and re.match(
                r"^(?: {2,}|\t)\S", lines[index + 1]
            ):
                index += 1
                continue
            break
        footnotes[identifier] = " ".join(part for part in parts if part).strip()
    return "\n".join(output), footnotes


ONES = [
    "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
    "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn",
    "siebzehn", "achtzehn", "neunzehn",
]
TENS = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"]
LARGE = [
    (10**21, "Trilliarde", "Trilliarden"),
    (10**18, "Trillion", "Trillionen"),
    (10**15, "Billiarde", "Billiarden"),
    (10**12, "Billion", "Billionen"),
    (10**9, "Milliarde", "Milliarden"),
    (10**6, "Million", "Millionen"),
]
QUANTITY_WORDS = {
    "Jahre", "Jahren", "Tonnen", "Menschen", "Kinder", "Kilometer", "Meter", "Personen",
}


def _under_hundred(number: int, final: bool = True) -> str:
    if number < 20:
        return "ein" if number == 1 and not final else ONES[number]
    ones, tens = number % 10, number // 10
    return (("ein" if ones == 1 else ONES[ones]) + "und" if ones else "") + TENS[tens]


def _under_thousand(number: int, final: bool = True) -> str:
    hundreds, rest = divmod(number, 100)
    prefix = (("ein" if hundreds == 1 else ONES[hundreds]) + "hundert") if hundreds else ""
    return prefix + (_under_hundred(rest, final) if rest else "")


def cardinal(number: int) -> str:
    if number == 0:
        return "null"
    if number < 0:
        return "minus " + cardinal(-number)
    remainder = number
    parts: list[str] = []
    for value, singular, plural in LARGE:
        if remainder >= value:
            count, remainder = divmod(remainder, value)
            parts.append("eine " + singular if count == 1 else cardinal(count) + " " + plural)
    if remainder:
        thousands, rest = divmod(remainder, 1000)
        word = ((_under_thousand(thousands, final=False) if thousands > 1 else "") + "tausend") if thousands else ""
        word += _under_thousand(rest) if rest else ""
        parts.append(word)
    return " ".join(parts)


def year_word(number: int) -> str:
    if 1100 <= number <= 1999:
        hundreds, rest = divmod(number, 100)
        return _under_hundred(hundreds) + "hundert" + (_under_hundred(rest) if rest else "")
    return cardinal(number)


LIST_ORDINALS = {
    1: "erstens", 2: "zweitens", 3: "drittens", 4: "viertens", 5: "fünftens",
    6: "sechstens", 7: "siebtens", 8: "achtens", 9: "neuntens", 10: "zehntens",
    11: "elftens", 12: "zwölftens", 13: "dreizehntens", 14: "vierzehntens",
    15: "fünfzehntens", 16: "sechzehntens", 17: "siebzehntens",
    18: "achtzehntens", 19: "neunzehntens", 20: "zwanzigstens",
}


def list_ordinal(number: int) -> str | None:
    if number in LIST_ORDINALS:
        return LIST_ORDINALS[number]
    if number == 100:
        return "hundertstens"
    if 20 < number < 100:
        tens, ones = divmod(number, 10)
        if not ones:
            return TENS[tens] + "stens"
        return ("ein" if ones == 1 else ONES[ones]) + "und" + TENS[tens] + "stens"
    return None


def normalize_numbered_lists(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = list_ordinal(int(match.group("number")))
        return match.group("indent") + ((value.capitalize() + ", ") if value else match.group(0).lstrip())

    return re.sub(
        r"(?m)^(?P<indent>[ \t]*)(?P<number>\d{1,3})[.)][ \t]+",
        replace,
        text,
    )


def numbers_to_words(text: str) -> str:
    text = re.sub(r"(?<=\d)[ .](?=\d{3}\b)", "", text)

    def replace_decimal(match: re.Match[str]) -> str:
        left = cardinal(int(match.group(1)))
        right = " ".join(ONES[int(digit)] for digit in match.group(2))
        return f"{left} Komma {right}"

    text = re.sub(r"(?<!\w)(\d+),(\d+)(?!\w)", replace_decimal, text)

    def replace_integer(match: re.Match[str]) -> str:
        digits = match.group("digits")
        number = int(digits)
        following = match.group("following") or ""
        following_word = following.strip().rstrip(".,;:!?")
        if len(digits) == 4 and 1100 <= number <= 2099 and following_word not in QUANTITY_WORDS:
            return year_word(number) + following
        return cardinal(number) + following

    return re.sub(
        r"(?<![\w.])(?P<digits>\d+)(?P<following>\s+[A-Za-zÄÖÜäöüß]+)?",
        replace_integer,
        text,
    )


NUMBER_PARTS = sorted(
    [
        "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn",
        "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig",
        "hundert", "tausend", "und", "null", "eins", "ein", "zwei", "drei", "vier", "fünf",
        "sechs", "sieben", "acht", "neun", "zehn", "elf", "zwölf",
    ],
    key=len,
    reverse=True,
)
NUMBER_WORD_RE = re.compile(
    r"((?:" + "|".join(NUMBER_PARTS) + r"){2,}?)(sten|ster|stes|ste|ten|ter|tes|te|jährigen|jährige|jährig)?",
    re.IGNORECASE,
)


def hyphenate_number_words(text: str) -> str:
    part_re = re.compile("|".join(NUMBER_PARTS), re.IGNORECASE)

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        parsed = NUMBER_WORD_RE.fullmatch(word)
        if not parsed:
            return word
        stem, ending = parsed.group(1), parsed.group(2) or ""
        parts = part_re.findall(stem)
        if "".join(parts).lower() != stem.lower() or sum(p.lower() != "und" for p in parts) < 2:
            return word
        return "-".join(parts) + (("-" + ending) if ending.startswith("jährig") else ending)

    return re.sub(r"[A-Za-zÄÖÜäöüß]+", replace, text)


GENERAL_REPLACEMENTS = (
    Replacement("z. B.", "zum Beispiel"),
    Replacement("z.B.", "zum Beispiel"),
    Replacement("d. h.", "das heißt"),
    Replacement("d.h.", "das heißt"),
    Replacement("u. a.", "unter anderem"),
    Replacement("u.a.", "unter anderem"),
    Replacement("ggf.", "gegebenenfalls"),
    Replacement("bzw.", "beziehungsweise"),
    Replacement("usw.", "und so weiter"),
)


def apply_literal_replacements(text: str, replacements: Sequence[Replacement]) -> str:
    for replacement in sorted(replacements, key=lambda item: len(item.source), reverse=True):
        if replacement.source:
            text = text.replace(replacement.source, replacement.target)
    return text


def apply_pronunciation(text: str, replacements: Sequence[Replacement]) -> str:
    for replacement in sorted(replacements, key=lambda item: len(item.source), reverse=True):
        if not replacement.source:
            continue
        pattern = r"(?<![\wÄÖÜäöüß-])" + re.escape(replacement.source) + r"(?![\wÄÖÜäöüß])"
        text = re.sub(pattern, replacement.target, text)
    return text


def _strip_inline_markdown(text: str) -> str:
    # Images are visual-only content.  Keeping their alt text caused generic OCR
    # placeholders such as "image" to be handed to the speech model.
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"!\[[^\]]*\]\[[^\]]*\]", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<https?://[^>]+>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"(?<!\\)(?:\*\*|__)(.+?)(?:\*\*|__)", r"\1", text)
    text = re.sub(r"(?<!\\)(?:\*|_)(.+?)(?:\*|_)", r"\1", text)
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    text = re.sub(r"\\([\\`*_{}\[\]()#+.!-])", r"\1", text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*?)?\s*/?>", "", text)
    text = re.sub(r"<![^>]*>", "", text)
    return html.unescape(text)


def strip_markdown(text: str) -> str:
    """Convert common Markdown constructs into readable plain paragraphs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1\s*>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)
    text = re.sub(r"^ {0,3}(```|~~~).*?^ {0,3}\1[^\n]*$", "", text, flags=re.DOTALL | re.MULTILINE)
    # Emphasis may span lines in OCR-generated Markdown.  Handle it before the
    # line-by-line cleanup so its delimiters cannot reach the speech model.
    text = re.sub(r"(?<!\\)\*\*(?![\s*])(.+?)(?<![\s*])\*\*(?!\*)", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\\)__(?![\s_])(.+?)(?<![\s_])__(?!_)", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<![\\*])\*(?![\s*])(.+?)(?<![\s*])\*(?!\*)", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<![\\_])_(?![\s_])(.+?)(?<![\s_])_(?!_)", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"^([^\n]+)\n=+[ \t]*$", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"^([^\n]+)\n-{3,}[ \t]*$", r"\1", text, flags=re.MULTILINE)
    output: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line
        if not line.strip():
            if output and output[-1] != "":
                output.append("")
            continue
        if re.match(r"^\s*(?:[-*_]\s*){3,}$", line):
            continue
        if re.match(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$", line):
            continue
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        line = re.sub(r"\s+#+\s*$", "", line)
        line = re.sub(r"^\s*(?:>\s?)+", "", line)
        line = re.sub(r"^\s*[-*+]\s+\[[ xX]\]\s*", "", line)
        line = re.sub(r"^\s*[-*+]\s+", "", line)
        if re.match(r"^\s*\|.*\|\s*$", line):
            line = line.strip().strip("|").replace(" | ", ", ")
        line = _strip_inline_markdown(line)
        line = re.sub(r"[ \t]{2,}", " ", line).strip()
        if line:
            output.append(line)
    while output and not output[-1]:
        output.pop()
    return "\n".join(output)


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n[ \t]*\n+", text) if part.strip()]


def _prepare_plain_text(text: str, options: SpeechOptions) -> str:
    text = apply_literal_replacements(text, options.project_replacements)
    text = apply_literal_replacements(text, GENERAL_REPLACEMENTS)
    text = normalize_numbered_lists(text)
    text = strip_markdown(text)
    text = text.replace(" = ", " gleich ")
    text = re.sub(r"\bOK\b", "okay", text)
    text = numbers_to_words(text)
    text = re.sub(r"\.{2,}|…[.…]*", "…", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    if options.hyphenate_number_words:
        text = hyphenate_number_words(text)
    text = apply_pronunciation(text, options.pronunciation_entries)
    return text.strip()


def prepare_segment_for_speech(
    segment_text: str,
    footnotes: dict[str, str],
    spoken_footnotes: set[str],
    options: SpeechOptions,
) -> str:
    """Create the exact TTS text for one segment."""
    without_definitions, _local_footnotes = extract_footnotes(segment_text)
    output: list[str] = []
    for paragraph in _paragraphs(without_definitions):
        references = [match.group("id") for match in FOOTNOTE_REFERENCE_RE.finditer(paragraph)]
        paragraph = FOOTNOTE_REFERENCE_RE.sub("", paragraph)
        prepared = _prepare_plain_text(paragraph, options)
        if prepared:
            output.append(prepared)
        if options.speak_footnotes:
            for identifier in references:
                if identifier in spoken_footnotes or identifier not in footnotes:
                    continue
                spoken_footnotes.add(identifier)
                note = _prepare_plain_text(footnotes[identifier], options)
                if note:
                    output.append("Anmerkung. " + note)
    return "\n\n".join(output).strip()


def prepare_segments_for_speech(
    full_document: str,
    segments: Sequence[Segment],
    options: SpeechOptions,
) -> list[str]:
    _main_text, footnotes = extract_footnotes(full_document)
    prepared: list[str] = []
    for segment in segments:
        # Each MP3 must remain self-contained.  A reference in a deselected or
        # earlier file must not suppress the note in this segment.
        value = prepare_segment_for_speech(segment.source_text, footnotes, set(), options)
        segment.speech_text = value
        prepared.append(value)
    return prepared


def _read_semicolon_csv(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        fields = {field.strip() for field in (reader.fieldnames or []) if field}
        rows = [{(key or "").strip(): (value or "").strip() for key, value in row.items()} for row in reader]
    return rows, fields


def load_pronunciation_csv(path: Path) -> list[Replacement]:
    rows, fields = _read_semicolon_csv(path)
    required = {"Schreibweise", "Aussprache", "Phonetisch"}
    if not required.issubset(fields):
        raise ValueError("Aussprache-CSV benötigt: Schreibweise;Aussprache;Phonetisch;Hinweis")
    entries: list[Replacement] = []
    for row in rows:
        source = row.get("Schreibweise", "")
        target = row.get("Phonetisch", "") or row.get("Aussprache", "")
        if source and target and source != target:
            entries.append(Replacement(source, target, row.get("Hinweis", "")))
    return entries


def load_project_replacements(path: Path) -> list[Replacement]:
    rows, fields = _read_semicolon_csv(path)
    if not {"Suche", "Ersatz"}.issubset(fields):
        raise ValueError("Projektregeln benötigen: Suche;Ersatz;Hinweis")
    return [
        Replacement(row.get("Suche", ""), row.get("Ersatz", ""), row.get("Hinweis", ""))
        for row in rows
        if row.get("Suche", "")
    ]


def create_batch_config(
    base_config_path: Path,
    destination: Path,
    *,
    speaker: Path,
    language: str,
    min_chunk_chars: int,
    max_chunk_chars: int,
) -> Path:
    if min_chunk_chars <= 0 or max_chunk_chars <= 0 or min_chunk_chars > max_chunk_chars:
        raise ValueError("Die minimale Chunk-Größe darf die maximale nicht überschreiten.")
    config = json.loads(base_config_path.read_text(encoding="utf-8"))
    config["speaker"] = str(speaker.resolve())
    config["language"] = language.strip()
    config["min_chunk_chars"] = min_chunk_chars
    config["max_chunk_chars"] = max_chunk_chars
    config["target_chunk_chars"] = (min_chunk_chars + max_chunk_chars) // 2
    config["clear_markdown"] = False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination
