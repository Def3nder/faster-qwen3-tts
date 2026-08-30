#!/usr/bin/env python3
"""
Generate long-form speech with Qwen3-TTS 1.7B and encode MP3 once at the end.

The input text can be provided directly or read from a UTF-8 text file:

    python generate/generate_mp3_with_embedding.py \
        --text "Hallo Welt" \
        --output output.mp3

    python generate/generate_mp3_with_embedding.py \
        --input input.txt

TTS settings are loaded from ``config.json`` next to this script by default.
Use ``--config`` to select a different configuration file.

The default ``semantic`` mode uses short paragraph-aware chunks and prefers
paragraph, heading, and topic boundaries.  A fixed short text preroll is
generated and discarded before every chunk to hide the model's startup
transient.  Two line breaks and a period are appended to each TTS request so
the model has explicit text context after the final spoken word.
The old fixed-size chunker remains available as ``legacy``. Lossless float WAV
parts are kept during generation and deleted after successful MP3 encoding by
default; set ``parts_directory_policy`` to ``keep`` to retain them.  For
non-streaming generation, eight codec frames from the preceding text chunk are
used as causal decode context by default.

Use ``--sentences N`` to create chunks containing exactly N complete sentences.
Only the final chunk may contain fewer sentences.
Use ``--characters N`` to override ``max_chunk_chars`` from the configuration.
Use ``--padding`` for the legacy sentence-padding experiment.  The production
end marker for each generated chunk is configured separately.
"""

import argparse
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple


MAX_ALLOWED_CHUNK_CHARS = 3000

DEFAULT_CONFIG: dict[str, Any] = {
    "speaker": "",
    "language": "Auto",
    "instruct": "",
    "model_path": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "device": "cuda:0",
    "model_device": "cuda",
    "dtype": "bfloat16",
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.05,
    "do_sample": True,
    "seed": 1337,
    "seed_strategy": "increment",
    "min_new_tokens": 2,
    "warmup_max_new_tokens": 20,
    "max_new_tokens": 768,
    "model_max_seq_len": 2048,
    "non_streaming_mode": False,
    "generation_api": "non_streaming",
    "codec_context_frames": 8,
    "mode": "semantic",
    "legacy_chunk_chars": 400,
    "target_chunk_chars": 340,
    "min_chunk_chars": 220,
    "max_chunk_chars": 520,
    "text_preroll_enabled": True,
    "text_preroll_sentence": (
        "Am frühen Morgen lag ein ruhiges Licht über der weiten Landschaft.\n\n."
    ),
    "text_preroll_search_window_ms": 1200,
    "text_preroll_min_pause_ms": 120,
    "text_preroll_lead_in_ms": 30,
    "append_chunk_end_padding": True,
    "chunk_end_padding_text": "\n\n.",
    "sentence_pause_ms": 220,
    "paragraph_pause_ms": 400,
    "topic_pause_ms": 650,
    "max_leading_silence_ms": 120,
    "max_trailing_silence_ms": 160,
    "silence_threshold_db": -50.0,
    "edge_fade_ms": 8,
    "crossfade_ms": 0,
    "loudness_match_max_db": 1.5,
    "save_wav_parts": True,
    "parts_directory_policy": "delete",
    "speak_numbered_lists": True,
    "ref_audio": "",
    "ref_text": "",
    "ref_text_file": "",
    "icl_append_silence_ms": 500,
}

DTYPES = {"bfloat16", "float16", "float32"}
MODES = {"legacy", "semantic", "semantic_icl"}
GENERATION_APIS = {"non_streaming", "streaming"}
SEED_STRATEGIES = {"fixed", "increment"}
PARTS_DIRECTORY_POLICIES = {"delete", "keep"}

COMMON_ABBREVIATIONS = {
    "abb",
    "abs",
    "bsp",
    "bzw",
    "ca",
    "d.h",
    "dr",
    "e.g",
    "etc",
    "ggf",
    "hr",
    "i.e",
    "jr",
    "mr",
    "mrs",
    "ms",
    "nr",
    "prof",
    "s",
    "sog",
    "sr",
    "str",
    "u.a",
    "u.ä",
    "usw",
    "vgl",
    "vs",
    "z.b",
    "z.t",
}

SENTENCE_CLOSERS = "\"'”’»)]}"


class TextChunk(NamedTuple):
    text: str
    boundary_after: str = "sentence"


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def character_limit(value: str) -> int:
    parsed = positive_int(value)
    if parsed > MAX_ALLOWED_CHUNK_CHARS:
        raise argparse.ArgumentTypeError(
            f"must be between 1 and {MAX_ALLOWED_CHUNK_CHARS}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CUDA-graphed TTS with a precomputed speaker embedding, written as MP3"
    )
    parser.add_argument(
        "--speaker",
        help="Path to speaker embedding (.pt from extract_speaker.py; overrides config)",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text to synthesize")
    source.add_argument(
        "--input",
        type=Path,
        help="UTF-8 text file containing the text to synthesize",
    )
    parser.add_argument(
        "--language",
        help="Language (English, French, German, Spanish, ...; overrides config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output MP3 path (default with --input: input path with .mp3 extension)",
    )
    parser.add_argument(
        "--model_path",
        help="Model path (overrides config)",
    )
    parser.add_argument(
        "--device",
        help="Device for the speaker embedding (overrides config)",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        help="Model dtype (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Base random seed (overrides config)",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODES),
        help="Chunk/prompt mode (overrides config)",
    )
    parser.add_argument(
        "--generation-api",
        choices=sorted(GENERATION_APIS),
        help="Generate whole chunks or collect the streaming API (overrides config)",
    )
    parser.add_argument(
        "--target-chars",
        type=character_limit,
        metavar="N",
        help="Target semantic chunk size (overrides config)",
    )
    parser.add_argument(
        "--ref-audio",
        type=Path,
        help="Reference audio for semantic_icl (overrides config)",
    )
    ref_text_source = parser.add_mutually_exclusive_group()
    ref_text_source.add_argument(
        "--ref-text",
        help="Exact reference transcript for semantic_icl (overrides config)",
    )
    ref_text_source.add_argument(
        "--ref-text-file",
        type=Path,
        help="UTF-8 reference transcript file for semantic_icl (overrides config)",
    )
    parser.add_argument(
        "--instruct",
        help="Optional style instruction; most reliable with semantic_icl",
    )
    parser.add_argument(
        "--sentences",
        type=positive_int,
        metavar="N",
        help=(
            "Create each chunk from exactly N complete sentences "
            "(the final chunk may contain fewer)"
        ),
    )
    parser.add_argument(
        "--characters",
        type=character_limit,
        metavar="N",
        help=(
            "Maximum characters per chunk; overrides max_chunk_chars from config "
            f"(1-{MAX_ALLOWED_CHUNK_CHARS})"
        ),
    )
    parser.add_argument(
        "--padding",
        action="store_true",
        help="Append two line breaks and a period to every complete sentence",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
        help="JSON configuration path (default: generate/config.json)",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        help="Write reproducible timing/chunk/VRAM metrics as JSON",
    )
    parser.add_argument(
        "--no-wav-parts",
        action="store_true",
        help="Do not retain per-chunk lossless WAV files",
    )
    return parser


def _require_type(config: dict[str, Any], key: str, expected: type) -> None:
    value = config[key]
    if expected is int:
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif expected is float:
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        valid = isinstance(value, expected)
    if not valid:
        raise ValueError(f"config value {key!r} must be {expected.__name__}")


def load_config(
    path: Path,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load, merge, apply CLI overrides, and validate the TTS configuration."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(raw, dict):
        raise ValueError("config must contain a JSON object")

    unknown = sorted(set(raw) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"unknown config value(s): {', '.join(unknown)}")

    speaker_from_cli = bool(
        cli_overrides and cli_overrides.get("speaker") is not None
    )
    config = {**DEFAULT_CONFIG, **raw}
    if cli_overrides:
        config.update(
            {
                key: value
                for key, value in cli_overrides.items()
                if value is not None
            }
        )

    for key in (
        "speaker",
        "language",
        "instruct",
        "model_path",
        "device",
        "model_device",
        "dtype",
        "seed_strategy",
        "generation_api",
        "mode",
        "parts_directory_policy",
        "ref_audio",
        "ref_text",
        "ref_text_file",
        "text_preroll_sentence",
        "chunk_end_padding_text",
    ):
        _require_type(config, key, str)
    for key in (
        "temperature",
        "top_p",
        "repetition_penalty",
        "silence_threshold_db",
        "loudness_match_max_db",
    ):
        _require_type(config, key, float)
    for key in (
        "top_k",
        "seed",
        "min_new_tokens",
        "warmup_max_new_tokens",
        "max_new_tokens",
        "model_max_seq_len",
        "codec_context_frames",
        "legacy_chunk_chars",
        "target_chunk_chars",
        "min_chunk_chars",
        "max_chunk_chars",
        "sentence_pause_ms",
        "paragraph_pause_ms",
        "topic_pause_ms",
        "max_leading_silence_ms",
        "max_trailing_silence_ms",
        "edge_fade_ms",
        "crossfade_ms",
        "icl_append_silence_ms",
        "text_preroll_search_window_ms",
        "text_preroll_min_pause_ms",
        "text_preroll_lead_in_ms",
    ):
        _require_type(config, key, int)
    for key in (
        "do_sample",
        "non_streaming_mode",
        "save_wav_parts",
        "speak_numbered_lists",
        "text_preroll_enabled",
        "append_chunk_end_padding",
    ):
        _require_type(config, key, bool)

    if not config["speaker"].strip():
        raise ValueError("config value 'speaker' must point to a .pt speaker embedding")
    if config["dtype"] not in DTYPES:
        raise ValueError(f"config value 'dtype' must be one of: {', '.join(sorted(DTYPES))}")
    if config["mode"] not in MODES:
        raise ValueError(f"config value 'mode' must be one of: {', '.join(sorted(MODES))}")
    if config["generation_api"] not in GENERATION_APIS:
        raise ValueError(
            "config value 'generation_api' must be one of: "
            f"{', '.join(sorted(GENERATION_APIS))}"
        )
    if config["seed_strategy"] not in SEED_STRATEGIES:
        raise ValueError(
            "config value 'seed_strategy' must be one of: "
            f"{', '.join(sorted(SEED_STRATEGIES))}"
        )
    if config["parts_directory_policy"] not in PARTS_DIRECTORY_POLICIES:
        raise ValueError(
            "config value 'parts_directory_policy' must be one of: "
            f"{', '.join(sorted(PARTS_DIRECTORY_POLICIES))}"
        )
    if config["temperature"] < 0:
        raise ValueError("config value 'temperature' must be >= 0")
    if config["top_k"] < 0:
        raise ValueError("config value 'top_k' must be >= 0")
    if not 0 < config["top_p"] <= 1:
        raise ValueError("config value 'top_p' must be in (0, 1]")
    if config["repetition_penalty"] <= 0:
        raise ValueError("config value 'repetition_penalty' must be > 0")
    if (
        config["min_new_tokens"] < 0
        or config["warmup_max_new_tokens"] <= 0
        or config["max_new_tokens"] <= 0
        or config["model_max_seq_len"] <= 0
    ):
        raise ValueError("token limits in config must be > 0")
    if config["model_max_seq_len"] <= config["max_new_tokens"]:
        raise ValueError(
            "config value 'model_max_seq_len' must be greater than "
            "'max_new_tokens' to leave room for the text prefill"
        )
    if config["codec_context_frames"] < 0:
        raise ValueError("config value 'codec_context_frames' must be >= 0")
    chunk_sizes = (
        config["legacy_chunk_chars"],
        config["min_chunk_chars"],
        config["target_chunk_chars"],
        config["max_chunk_chars"],
    )
    if any(not 1 <= value <= MAX_ALLOWED_CHUNK_CHARS for value in chunk_sizes):
        raise ValueError(
            "chunk sizes must be between 1 and " f"{MAX_ALLOWED_CHUNK_CHARS}"
        )
    if not (
        config["min_chunk_chars"]
        <= config["target_chunk_chars"]
        <= config["max_chunk_chars"]
    ):
        raise ValueError(
            "semantic chunk sizes must satisfy min_chunk_chars <= "
            "target_chunk_chars <= max_chunk_chars"
        )
    millisecond_keys = (
        "sentence_pause_ms",
        "paragraph_pause_ms",
        "topic_pause_ms",
        "max_leading_silence_ms",
        "max_trailing_silence_ms",
        "edge_fade_ms",
        "crossfade_ms",
        "icl_append_silence_ms",
    )
    if any(config[key] < 0 for key in millisecond_keys):
        raise ValueError("pause, fade, trim, and ICL silence values must be >= 0")
    if config["loudness_match_max_db"] < 0:
        raise ValueError("config value 'loudness_match_max_db' must be >= 0")
    if config["text_preroll_enabled"]:
        if config["generation_api"] != "non_streaming":
            raise ValueError(
                "text preroll requires generation_api 'non_streaming'"
            )
        if not config["text_preroll_sentence"].strip():
            raise ValueError(
                "config value 'text_preroll_sentence' must not be empty"
            )
        if config["text_preroll_min_pause_ms"] <= 0:
            raise ValueError(
                "config value 'text_preroll_min_pause_ms' must be > 0"
            )
        if (
            config["text_preroll_search_window_ms"] < 0
            or config["text_preroll_lead_in_ms"] < 0
        ):
            raise ValueError(
                "text preroll search window and lead-in must be >= 0"
            )
    if config["append_chunk_end_padding"]:
        if config["chunk_end_padding_text"] != "\n\n.":
            raise ValueError(
                "config value 'chunk_end_padding_text' must be exactly '\\n\\n.'"
            )

    speaker_path = Path(config["speaker"]).expanduser()
    if not speaker_path.is_absolute():
        base_path = Path.cwd() if speaker_from_cli else path.resolve().parent
        speaker_path = (base_path / speaker_path).resolve()
    if not speaker_path.is_file() and not speaker_path.suffix:
        model_size = "1.7B" if "1.7B" in config["model_path"] else "0.6B"
        candidates = (
            speaker_path.with_name(f"{speaker_path.name}-{model_size}.pt"),
            speaker_path.with_suffix(".pt"),
        )
        speaker_path = next((candidate for candidate in candidates if candidate.is_file()), speaker_path)
    if not speaker_path.is_file():
        raise ValueError(f"speaker embedding not found: {speaker_path}")
    config["speaker"] = speaker_path

    for key in ("ref_audio", "ref_text_file"):
        value = config[key].strip()
        if not value:
            config[key] = None
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = (path.resolve().parent / candidate).resolve()
        if not candidate.is_file():
            raise ValueError(f"config value {key!r} does not exist: {candidate}")
        config[key] = candidate

    if config["ref_text_file"] is not None:
        config["ref_text"] = config["ref_text_file"].read_text(
            encoding="utf-8-sig"
        ).strip()
    if config["mode"] == "semantic_icl" and (
        config["ref_audio"] is None or not config["ref_text"].strip()
    ):
        raise ValueError(
            "semantic_icl requires both ref_audio and an exact ref_text or ref_text_file"
        )

    return config


def resolve_text(direct_text: str | None, input_path: Path | None) -> str:
    """Return non-empty input text from the selected CLI source."""
    if direct_text is not None:
        text = direct_text
    else:
        assert input_path is not None
        try:
            text = input_path.read_text(encoding="utf-8-sig")
        except FileNotFoundError as exc:
            raise ValueError(f"input text file not found: {input_path}") from exc
        except UnicodeDecodeError as exc:
            raise ValueError(f"input text file is not valid UTF-8: {input_path}") from exc

    text = text.strip()
    if not text:
        raise ValueError("input text must not be empty")
    return text


GERMAN_LIST_ORDINALS = {
    1: "erstens",
    2: "zweitens",
    3: "drittens",
    4: "viertens",
    5: "fünftens",
    6: "sechstens",
    7: "siebtens",
    8: "achtens",
    9: "neuntens",
    10: "zehntens",
    11: "elftens",
    12: "zwölftens",
    13: "dreizehntens",
    14: "vierzehntens",
    15: "fünfzehntens",
    16: "sechzehntens",
    17: "siebzehntens",
    18: "achtzehntens",
    19: "neunzehntens",
}
GERMAN_CARDINAL_ONES = {
    1: "ein",
    2: "zwei",
    3: "drei",
    4: "vier",
    5: "fünf",
    6: "sechs",
    7: "sieben",
    8: "acht",
    9: "neun",
}
GERMAN_CARDINAL_TENS = {
    2: "zwanzig",
    3: "dreißig",
    4: "vierzig",
    5: "fünfzig",
    6: "sechzig",
    7: "siebzig",
    8: "achtzig",
    9: "neunzig",
}


def _german_list_ordinal(number: int) -> str | None:
    """Return a German ordinal adverb suitable for a spoken list marker."""
    if number in GERMAN_LIST_ORDINALS:
        return GERMAN_LIST_ORDINALS[number]
    if number == 100:
        return "hundertstens"
    if not 20 <= number <= 99:
        return None

    tens, ones = divmod(number, 10)
    tens_word = GERMAN_CARDINAL_TENS[tens]
    if ones == 0:
        return f"{tens_word}stens"
    return f"{GERMAN_CARDINAL_ONES[ones]}und{tens_word}stens"


def normalize_numbered_list_markers(text: str, language: str) -> str:
    """Turn German line-leading ``1.`` list markers into ``Erstens,``."""
    if language.strip().casefold() not in {"de", "de-de", "deutsch", "german"}:
        return text

    def replace_marker(match: re.Match[str]) -> str:
        ordinal = _german_list_ordinal(int(match.group("number")))
        if ordinal is None:
            return match.group(0)
        return f"{match.group('indent')}{ordinal.capitalize()}, "

    return re.sub(
        r"(?m)^(?P<indent>[ \t]*)(?P<number>\d{1,3})\.[ \t]+(?=\S)",
        replace_marker,
        text,
    )


def _period_ends_abbreviation(text: str, period_index: int) -> bool:
    token_match = re.search(r"(\S+)$", text[: period_index + 1])
    if token_match is None:
        return False

    token = token_match.group(1).lstrip("\"'“‘»([{").lower()
    if not token.endswith("."):
        return False

    without_period = token[:-1]
    if without_period in COMMON_ABBREVIATIONS:
        return True
    if re.fullmatch(r"(?:[a-zäöü]\.){2,}", token):
        return True
    if re.fullmatch(r"[a-zäöü]\.", token):
        return True

    line_start = text.rfind("\n", 0, period_index) + 1
    line_prefix = text[line_start : period_index + 1].strip()
    return bool(re.fullmatch(r"\d+\.", line_prefix))


def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph conservatively at sentence-ending punctuation."""
    sentences: list[str] = []
    sentence_start = 0
    index = 0

    while index < len(paragraph):
        if paragraph[index] not in ".!?":
            index += 1
            continue

        punctuation_start = index
        while index < len(paragraph) and paragraph[index] in ".!?":
            index += 1
        punctuation = paragraph[punctuation_start:index]

        sentence_end = index
        while sentence_end < len(paragraph) and paragraph[sentence_end] in SENTENCE_CLOSERS:
            sentence_end += 1

        if sentence_end < len(paragraph) and not paragraph[sentence_end].isspace():
            index = sentence_end
            continue

        if (
            punctuation == "."
            and _period_ends_abbreviation(paragraph, punctuation_start)
        ):
            index = sentence_end
            continue

        sentence = paragraph[sentence_start:sentence_end].strip()
        if sentence:
            sentences.append(sentence)
        sentence_start = sentence_end
        while sentence_start < len(paragraph) and paragraph[sentence_start].isspace():
            sentence_start += 1
        index = sentence_start

    remainder = paragraph[sentence_start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def _chunk_by_sentence_count(
    paragraphs: list[str],
    sentences_per_chunk: int,
    max_chars: int,
    padding: bool,
) -> list[str]:
    sentence_units: list[tuple[str, int]] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        for sentence in split_sentences(paragraph):
            sentence = _prepare_sentence(sentence, padding)
            if len(sentence) > max_chars:
                preview = sentence[:80].replace("\n", " ")
                raise ValueError(
                    f"single sentence has {len(sentence)} characters and exceeds "
                    f"the {max_chars}-character chunk limit: {preview!r}"
                )
            sentence_units.append((sentence, paragraph_index))

    chunks: list[str] = []
    for start in range(0, len(sentence_units), sentences_per_chunk):
        batch = sentence_units[start : start + sentences_per_chunk]
        chunk = ""
        previous_paragraph: int | None = None
        for sentence, paragraph_index in batch:
            if not chunk:
                chunk = sentence
            else:
                separator = (
                    "\n\n"
                    if paragraph_index != previous_paragraph
                    else " "
                )
                chunk = f"{chunk}{separator}{sentence}"
            previous_paragraph = paragraph_index

        if len(chunk) > max_chars:
            chunk_number = len(chunks) + 1
            raise ValueError(
                f"--sentences {sentences_per_chunk} produces chunk "
                f"{chunk_number} with {len(chunk)} characters, exceeding the "
                f"{max_chars}-character limit; reduce --sentences"
            )
        chunks.append(chunk)
    return chunks


def _prepare_sentence(sentence: str, padding: bool) -> str:
    return f"{sentence}\n\n." if padding else sentence


def _prepare_paragraph(paragraph: str, padding: bool) -> str:
    if not padding:
        return paragraph
    return " ".join(
        _prepare_sentence(sentence, True)
        for sentence in split_sentences(paragraph)
    )


def chunk_text(
    text: str,
    max_chars: int = MAX_ALLOWED_CHUNK_CHARS,
    sentences_per_chunk: int | None = None,
    padding: bool = False,
) -> list[str]:
    """Chunk text by whole paragraphs, falling back to whole sentences."""
    if not 1 <= max_chars <= MAX_ALLOWED_CHUNK_CHARS:
        raise ValueError(
            f"max_chars must be between 1 and {MAX_ALLOWED_CHUNK_CHARS}"
        )

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError("input text must not be empty")

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n[ \t]*\n+", normalized)
        if paragraph.strip()
    ]
    if sentences_per_chunk is not None:
        if (
            not isinstance(sentences_per_chunk, int)
            or isinstance(sentences_per_chunk, bool)
            or sentences_per_chunk <= 0
        ):
            raise ValueError("sentences_per_chunk must be a positive integer")
        return _chunk_by_sentence_count(
            paragraphs,
            sentences_per_chunk,
            max_chars,
            padding,
        )

    chunks: list[str] = []
    current_chunk = ""

    def flush_current() -> None:
        nonlocal current_chunk
        if current_chunk:
            chunks.append(current_chunk)
            current_chunk = ""

    for paragraph in paragraphs:
        prepared_paragraph = _prepare_paragraph(paragraph, padding)
        if len(prepared_paragraph) <= max_chars:
            candidate = (
                f"{current_chunk}\n\n{prepared_paragraph}"
                if current_chunk
                else prepared_paragraph
            )
            if len(candidate) <= max_chars:
                current_chunk = candidate
            else:
                flush_current()
                current_chunk = prepared_paragraph
            continue

        flush_current()
        sentence_chunk = ""
        for sentence in split_sentences(paragraph):
            sentence = _prepare_sentence(sentence, padding)
            if len(sentence) > max_chars:
                preview = sentence[:80].replace("\n", " ")
                raise ValueError(
                    f"single sentence has {len(sentence)} characters and exceeds "
                    f"the {max_chars}-character chunk limit: {preview!r}"
                )

            candidate = (
                f"{sentence_chunk} {sentence}"
                if sentence_chunk
                else sentence
            )
            if len(candidate) <= max_chars:
                sentence_chunk = candidate
            else:
                chunks.append(sentence_chunk)
                sentence_chunk = sentence
        if sentence_chunk:
            chunks.append(sentence_chunk)

    flush_current()

    if not chunks:
        raise ValueError("input text must not be empty")
    if any(len(chunk) > max_chars for chunk in chunks):
        raise RuntimeError("internal error: generated chunk exceeds character limit")
    return chunks


class _SemanticUnit(NamedTuple):
    text: str
    block_index: int
    boundary_after: str


def _looks_like_heading(text: str) -> bool:
    stripped = text.strip()
    if re.match(r"^#{1,6}\s+\S", stripped):
        return True
    if "\n" in stripped or len(stripped) > 120:
        return False
    if re.match(r"^\d+(?:\.\d+)*[.)]?\s+\S", stripped):
        return not bool(re.search(r"[.!?][\"'”’»)]?$", stripped))
    word_count = len(stripped.split())
    return (
        not bool(re.search(r"[.!?][\"'”’»)]?$", stripped))
        and 4 <= word_count <= 12
    )


def _spoken_block_text(text: str) -> str:
    """Remove Markdown heading syntax without rewriting spoken content."""
    return re.sub(r"^#{1,6}\s+", "", text.strip())


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Split an unavoidable overlong sentence at a clause or word boundary."""
    parts: list[str] = []
    remainder = sentence.strip()
    while len(remainder) > max_chars:
        window = remainder[: max_chars + 1]
        search_from = max(1, int(max_chars * 0.55))
        split_at = -1
        for match in re.finditer(r"[;:,–—]\s+", window[search_from:]):
            split_at = search_from + match.end()
        if split_at < 0:
            split_at = window.rfind(" ", search_from)
        if split_at <= 0:
            raise ValueError(
                f"cannot split a {len(sentence)}-character sentence below "
                f"max_chunk_chars={max_chars} without breaking a word"
            )
        part = remainder[:split_at].strip()
        if not part:
            raise RuntimeError("internal error while splitting a long sentence")
        parts.append(part)
        remainder = remainder[split_at:].strip()
    if remainder:
        parts.append(remainder)
    return parts


def _semantic_units(text: str, max_chars: int) -> list[_SemanticUnit]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    blocks = [
        block.strip()
        for block in re.split(r"\n[ \t]*\n+", normalized)
        if block.strip()
    ]
    units: list[_SemanticUnit] = []
    for block_index, raw_block in enumerate(blocks):
        block = _spoken_block_text(raw_block)
        sentences = split_sentences(block) or [block]
        expanded: list[str] = []
        for sentence in sentences:
            expanded.extend(_split_long_sentence(sentence, max_chars))

        next_is_heading = (
            block_index + 1 < len(blocks)
            and _looks_like_heading(blocks[block_index + 1])
        )
        for sentence_index, sentence in enumerate(expanded):
            is_last = sentence_index == len(expanded) - 1
            if not is_last:
                boundary = "sentence"
            elif next_is_heading:
                boundary = "topic"
            elif block_index + 1 < len(blocks):
                boundary = "paragraph"
            else:
                boundary = "end"
            units.append(_SemanticUnit(sentence, block_index, boundary))
    return units


def _join_semantic_units(units: list[_SemanticUnit], start: int, end: int) -> str:
    pieces: list[str] = []
    previous_block: int | None = None
    for unit in units[start:end]:
        if pieces:
            pieces.append("\n\n" if unit.block_index != previous_block else " ")
        pieces.append(unit.text)
        previous_block = unit.block_index
    return "".join(pieces)


def semantic_chunk_text(
    text: str,
    *,
    target_chars: int = 1000,
    min_chars: int = 700,
    max_chars: int = 1300,
) -> list[TextChunk]:
    """Pack natural text units near a target size using global break selection."""
    if not 1 <= min_chars <= target_chars <= max_chars <= MAX_ALLOWED_CHUNK_CHARS:
        raise ValueError(
            "semantic sizes must satisfy 1 <= min_chars <= target_chars <= "
            f"max_chars <= {MAX_ALLOWED_CHUNK_CHARS}"
        )
    if not text.strip():
        raise ValueError("input text must not be empty")

    units = _semantic_units(text, max_chars)
    if not units:
        raise ValueError("input text must not be empty")

    count = len(units)
    costs = [math.inf] * (count + 1)
    choices = [count] * (count + 1)
    costs[count] = 0.0
    boundary_bonus = {"sentence": 0.0, "paragraph": 0.35, "topic": 1.0, "end": 0.5}

    for start in range(count - 1, -1, -1):
        for end in range(start + 1, count + 1):
            if any(
                units[index].boundary_after == "topic"
                for index in range(start, end - 1)
            ):
                break
            chunk = _join_semantic_units(units, start, end)
            length = len(chunk)
            if length > max_chars:
                break
            boundary = units[end - 1].boundary_after
            size_cost = abs(length - target_chars) / target_chars
            if length < min_chars:
                underweight = 1.25 if end == count else 4.0
                size_cost += underweight * (min_chars - length) / min_chars
            total_cost = size_cost - boundary_bonus[boundary] + costs[end]
            if total_cost < costs[start]:
                costs[start] = total_cost
                choices[start] = end

    if math.isinf(costs[0]):
        raise RuntimeError("unable to find a semantic chunking solution")

    chunks: list[TextChunk] = []
    start = 0
    while start < count:
        end = choices[start]
        if end <= start:
            raise RuntimeError("internal semantic chunking error")
        chunks.append(
            TextChunk(
                _join_semantic_units(units, start, end),
                units[end - 1].boundary_after,
            )
        )
        start = end
    return chunks


def build_text_chunks(
    text: str,
    config: dict[str, Any],
    *,
    sentences_per_chunk: int | None = None,
    padding: bool = False,
) -> list[TextChunk]:
    spoken_text = (
        normalize_numbered_list_markers(text, config["language"])
        if config["speak_numbered_lists"]
        else text
    )
    if config["mode"] in {"semantic", "semantic_icl"}:
        if sentences_per_chunk is not None:
            raise ValueError("--sentences is only supported in legacy mode")
        if padding:
            raise ValueError("--padding is only supported in legacy mode")
        return semantic_chunk_text(
            spoken_text,
            target_chars=config["target_chunk_chars"],
            min_chars=config["min_chunk_chars"],
            max_chars=config["max_chunk_chars"],
        )

    legacy = chunk_text(
        spoken_text,
        max_chars=config["legacy_chunk_chars"],
        sentences_per_chunk=sentences_per_chunk,
        padding=padding,
    )
    paragraph_endings = {
        paragraph.strip()
        for paragraph in re.split(r"\n[ \t]*\n+", spoken_text.strip())
        if paragraph.strip()
    }
    result: list[TextChunk] = []
    for index, chunk in enumerate(legacy):
        if index == len(legacy) - 1:
            boundary = "end"
        elif any(chunk.endswith(ending) for ending in paragraph_endings):
            boundary = "paragraph"
        else:
            boundary = "sentence"
        result.append(TextChunk(chunk, boundary))
    return result


def validate_output_path(path: Path) -> Path:
    if path.suffix.lower() != ".mp3":
        raise ValueError("--output must use the .mp3 file extension")
    return path


def resolve_output_path(output_path: Path | None, input_path: Path | None) -> Path:
    """Resolve an explicit output or derive it from the input text file."""
    if output_path is None:
        if input_path is None:
            raise ValueError("--output is required when --text is used")
        output_path = input_path.with_suffix(".mp3")
    return validate_output_path(output_path)


def load_xvector_prompt(path: Path, device: str) -> dict[str, list[Any]]:
    """Load a saved x-vector and return a voice_clone_prompt dict."""
    import torch

    spk_emb = torch.load(path, weights_only=True).to(device)
    if not isinstance(spk_emb, torch.Tensor) or spk_emb.numel() != 2048:
        raise ValueError(
            f"expected a 2048-element speaker embedding in {path}, "
            f"got {type(spk_emb).__name__} with shape {getattr(spk_emb, 'shape', None)}"
        )
    return {
        "ref_code": [None],
        "ref_spk_embedding": [spk_emb],
        "x_vector_only_mode": [True],
        "icl_mode": [False],
    }


def load_voice_prompt(model: Any, config: dict[str, Any]) -> Any:
    """Load the compact x-vector or build one reusable full ICL prompt."""
    if config["mode"] != "semantic_icl":
        return load_xvector_prompt(config["speaker"], device=config["device"])

    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(
        str(config["ref_audio"]), dtype="float32", always_2d=False
    )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    silence_samples = round(config["icl_append_silence_ms"] * sample_rate / 1000)
    if silence_samples:
        audio = np.concatenate(
            [audio, np.zeros(silence_samples, dtype=np.float32)]
        )
    return model.model.create_voice_clone_prompt(
        ref_audio=(audio, sample_rate),
        ref_text=config["ref_text"],
        x_vector_only_mode=False,
    )


def write_mp3(path: Path, audio: Any, sample_rate: int) -> None:
    """Write MPEG Layer III explicitly, independent of extension inference."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(
            path,
            audio,
            sample_rate,
            format="MP3",
            subtype="MPEG_LAYER_III",
        )
    except (RuntimeError, TypeError) as exc:
        raise RuntimeError(
            "MP3 encoding failed. Install a SoundFile/libsndfile build with MP3 support."
        ) from exc


def delete_parts_directory(parts_directory: Path, output: Path) -> None:
    """Delete only the exact, non-symlinked parts directory for ``output``."""
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError(
            f"refusing to delete WAV parts without a completed MP3: {output}"
        )
    expected = output.with_name(f"{output.stem}_parts")
    if parts_directory.resolve() != expected.resolve():
        raise RuntimeError(
            f"refusing to delete unexpected parts directory: {parts_directory}"
        )
    if parts_directory.is_symlink():
        raise RuntimeError(
            f"refusing to delete symlinked parts directory: {parts_directory}"
        )
    if not parts_directory.exists():
        return
    if not parts_directory.is_dir():
        raise RuntimeError(f"parts path is not a directory: {parts_directory}")
    shutil.rmtree(parts_directory)


def write_wav(path: Path, audio: Any, sample_rate: int) -> None:
    """Write a lossless float PCM intermediate without quantizing it."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, format="WAV", subtype="FLOAT")


def _to_float32_audio(audio: Any) -> Any:
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _silence_threshold(audio: Any, configured_db: float) -> float:
    import numpy as np

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak <= 0:
        return 10 ** (configured_db / 20)
    relative_db = 20 * math.log10(peak) - 45.0
    return 10 ** (max(configured_db, relative_db) / 20)


def edge_silence_samples(audio: Any, threshold_db: float) -> tuple[int, int]:
    import numpy as np

    array = _to_float32_audio(audio)
    if not len(array):
        return 0, 0
    active = np.flatnonzero(np.abs(array) > _silence_threshold(array, threshold_db))
    if not len(active):
        return len(array), len(array)
    return int(active[0]), int(len(array) - active[-1] - 1)


def trim_excess_edge_silence(
    audio: Any,
    sample_rate: int,
    *,
    threshold_db: float,
    max_leading_ms: int,
    max_trailing_ms: int,
) -> Any:
    """Cap only excessive edge silence while preserving breaths and room tone."""
    array = _to_float32_audio(audio)
    leading, trailing = edge_silence_samples(array, threshold_db)
    keep_leading = round(max_leading_ms * sample_rate / 1000)
    keep_trailing = round(max_trailing_ms * sample_rate / 1000)
    start = max(0, leading - keep_leading)
    end = len(array) - max(0, trailing - keep_trailing)
    return array[start:max(start, end)]


def _active_rms(audio: Any, threshold_db: float) -> float:
    import numpy as np

    array = _to_float32_audio(audio)
    if not len(array):
        return 0.0
    threshold = _silence_threshold(array, threshold_db)
    active = array[np.abs(array) > threshold]
    if not len(active):
        return 0.0
    return float(np.sqrt(np.mean(active.astype(np.float64) ** 2)))


def match_chunk_loudness(
    audio: Any,
    reference_rms: float | None,
    *,
    threshold_db: float,
    max_adjustment_db: float,
) -> tuple[Any, float]:
    """Apply a deliberately small RMS correction and return the observed RMS."""
    import numpy as np

    array = _to_float32_audio(audio)
    rms = _active_rms(array, threshold_db)
    if reference_rms is None or reference_rms <= 0 or rms <= 0 or max_adjustment_db == 0:
        return array, rms
    requested_db = 20 * math.log10(reference_rms / rms)
    applied_db = max(-max_adjustment_db, min(max_adjustment_db, requested_db))
    gain = 10 ** (applied_db / 20)
    adjusted = np.clip(array * gain, -1.0, 1.0).astype(np.float32)
    return adjusted, _active_rms(adjusted, threshold_db)


def _apply_edge_fades(audio: Any, sample_rate: int, fade_ms: int) -> Any:
    import numpy as np

    array = _to_float32_audio(audio).copy()
    fade_samples = min(round(fade_ms * sample_rate / 1000), len(array) // 2)
    if fade_samples <= 0:
        return array
    ramp = np.linspace(0.0, 1.0, fade_samples, endpoint=True, dtype=np.float32)
    array[:fade_samples] *= ramp
    array[-fade_samples:] *= ramp[::-1]
    return array


def join_audio_chunks(
    audio_chunks: list[Any],
    text_chunks: list[TextChunk],
    sample_rate: int,
    config: dict[str, Any],
) -> Any:
    """Join PCM once, avoiding doubled silence and unsafe text/audio overlap."""
    import numpy as np

    if not audio_chunks:
        return np.zeros(0, dtype=np.float32)
    pause_by_boundary = {
        "sentence": config["sentence_pause_ms"],
        "paragraph": config["paragraph_pause_ms"],
        "topic": config["topic_pause_ms"],
        "end": 0,
    }
    result = _apply_edge_fades(audio_chunks[0], sample_rate, config["edge_fade_ms"])
    for index, next_audio in enumerate(audio_chunks[1:], start=1):
        next_array = _apply_edge_fades(
            next_audio, sample_rate, config["edge_fade_ms"]
        )
        target_pause = round(
            pause_by_boundary[text_chunks[index - 1].boundary_after]
            * sample_rate
            / 1000
        )
        _, trailing = edge_silence_samples(result, config["silence_threshold_db"])
        leading, _ = edge_silence_samples(next_array, config["silence_threshold_db"])
        missing_pause = max(0, target_pause - trailing - leading)

        crossfade = min(
            round(config["crossfade_ms"] * sample_rate / 1000),
            len(result),
            len(next_array),
        )
        if crossfade > 0 and target_pause == 0:
            fade_out = np.linspace(1.0, 0.0, crossfade, dtype=np.float32)
            fade_in = 1.0 - fade_out
            overlap = result[-crossfade:] * fade_out + next_array[:crossfade] * fade_in
            result = np.concatenate([result[:-crossfade], overlap, next_array[crossfade:]])
        else:
            result = np.concatenate(
                [result, np.zeros(missing_pause, dtype=np.float32), next_array]
            )
    return result.astype(np.float32, copy=False)


def _generate_audio_chunk(
    model: Any,
    voice_clone_prompt: Any,
    text: str,
    config: dict[str, Any],
    fast_generate: Any,
    *,
    run_warmup: bool,
    seed: int,
    codec_context_state: dict[str, Any] | None = None,
) -> tuple[Any, int, dict[str, Any]]:
    import torch

    warmup_seconds = 0.0
    if run_warmup:
        warmup_start = time.perf_counter()
        # The captured graph is position independent; the real prefill is copied
        # into StaticCache for every chunk after capture.
        model.warmup(prefill_len=100)
        warmup_seconds += time.perf_counter() - warmup_start

    prepare_start = time.perf_counter()
    m, talker, talker_config, tie, tam, tth, tpe, ref_codes = model._prepare_generation(
        text=text,
        language=config["language"],
        ref_text=config["ref_text"],
        non_streaming_mode=config["non_streaming_mode"],
        voice_clone_prompt=voice_clone_prompt,
        instruct=config["instruct"] or None,
    )
    prepare_seconds = time.perf_counter() - prepare_start
    print(f"Prefill length: {tie.shape[1]} tokens")

    if run_warmup:
        warmup_start = time.perf_counter()
        talker.rope_deltas = None
        fast_generate(
            talker,
            tie,
            tam,
            tth,
            tpe,
            talker_config,
            model.predictor_graph,
            model.talker_graph,
            temperature=config["temperature"],
            top_k=config["top_k"],
            top_p=config["top_p"],
            do_sample=config["do_sample"],
            repetition_penalty=config["repetition_penalty"],
            min_new_tokens=config["min_new_tokens"],
            max_new_tokens=config["warmup_max_new_tokens"],
        )
        warmup_seconds += time.perf_counter() - warmup_start

    torch.manual_seed(seed)
    generation_start = time.perf_counter()
    talker.rope_deltas = None
    codec_ids, timing = fast_generate(
        talker,
        tie,
        tam,
        tth,
        tpe,
        talker_config,
        model.predictor_graph,
        model.talker_graph,
        temperature=config["temperature"],
        top_k=config["top_k"],
        top_p=config["top_p"],
        do_sample=config["do_sample"],
        repetition_penalty=config["repetition_penalty"],
        min_new_tokens=config["min_new_tokens"],
        max_new_tokens=config["max_new_tokens"],
    )
    generation_wall_seconds = time.perf_counter() - generation_start
    if codec_ids is None or codec_ids.numel() == 0:
        raise RuntimeError("generation returned no tokens")

    decode_start = time.perf_counter()
    prefix_parts = []
    if ref_codes is not None:
        prefix_parts.append(ref_codes.to(codec_ids.device))

    carried_frames = 0
    context_frames = config["codec_context_frames"]
    previous_codes = (
        codec_context_state.get("codes")
        if codec_context_state is not None and context_frames > 0
        else None
    )
    if previous_codes is not None:
        previous_codes = previous_codes.to(codec_ids.device)
        prefix_parts.append(previous_codes)
        carried_frames = int(previous_codes.shape[0])

    decode_parts = [*prefix_parts, codec_ids]
    codes_for_decode = (
        torch.cat(decode_parts, dim=0) if len(decode_parts) > 1 else codec_ids
    )
    prepended_frames = int(codes_for_decode.shape[0] - codec_ids.shape[0])
    wavs, sample_rate = m.speech_tokenizer.decode(
        {"audio_codes": codes_for_decode.unsqueeze(0)}
    )
    audio = _to_float32_audio(wavs[0])
    if prepended_frames > 0:
        cut = int(
            prepended_frames / max(int(codes_for_decode.shape[0]), 1) * len(audio)
        )
        audio = audio[cut:]
    codec_decode_seconds = time.perf_counter() - decode_start

    if codec_context_state is not None:
        codec_context_state["codes"] = (
            codec_ids[-context_frames:].detach().clone()
            if context_frames > 0
            else None
        )
    if context_frames > 0 and codec_context_state is not None:
        if carried_frames:
            print(
                f"Codec context: {carried_frames} frame(s) from previous chunk "
                "prepended and trimmed"
            )
        else:
            print("Codec context: cold start for first chunk")
    timing.update(
        {
            "prepare_s": prepare_seconds,
            "generation_wall_s": generation_wall_seconds,
            "codec_decode_s": codec_decode_seconds,
            "warmup_s": warmup_seconds,
            "codec_context_frames": carried_frames,
            "tts_wall_s": prepare_seconds
            + generation_wall_seconds
            + codec_decode_seconds,
        }
    )
    return audio, sample_rate, timing


def _generate_audio_chunk_streaming(
    model: Any,
    voice_clone_prompt: Any,
    text: str,
    config: dict[str, Any],
    *,
    run_warmup: bool,
    seed: int,
) -> tuple[Any, int, dict[str, Any]]:
    """Collect the streaming API for an apples-to-apples finished-file test."""
    import numpy as np
    import torch

    warmup_seconds = 0.0
    if run_warmup:
        warmup_start = time.perf_counter()
        model.warmup(prefill_len=100)
        warmup_seconds = time.perf_counter() - warmup_start

    torch.manual_seed(seed)
    start = time.perf_counter()
    audio_parts: list[Any] = []
    sample_rate: int | None = None
    last_timing: dict[str, Any] = {}
    stream = model.generate_voice_clone_streaming(
        text=text,
        language=config["language"],
        ref_text=config["ref_text"],
        voice_clone_prompt=voice_clone_prompt,
        non_streaming_mode=config["non_streaming_mode"],
        instruct=config["instruct"] or None,
        temperature=config["temperature"],
        top_k=config["top_k"],
        top_p=config["top_p"],
        do_sample=config["do_sample"],
        repetition_penalty=config["repetition_penalty"],
        min_new_tokens=config["min_new_tokens"],
        max_new_tokens=config["max_new_tokens"],
    )
    for audio, chunk_sample_rate, timing in stream:
        if sample_rate is None:
            sample_rate = chunk_sample_rate
        elif sample_rate != chunk_sample_rate:
            raise RuntimeError("sample rate changed inside streaming generation")
        audio_parts.append(_to_float32_audio(audio))
        last_timing = timing
    wall_seconds = time.perf_counter() - start
    if sample_rate is None or not audio_parts:
        raise RuntimeError("streaming generation returned no audio")
    combined = np.concatenate(audio_parts)
    timing = {
        **last_timing,
        "prepare_s": 0.0,
        "generation_wall_s": wall_seconds,
        "codec_decode_s": 0.0,
        "warmup_s": warmup_seconds,
        "tts_wall_s": wall_seconds,
        "steps": int(last_timing.get("steps", 0)),
        "prefill_ms": float(last_timing.get("prefill_ms", 0.0)),
        "decode_s": float(last_timing.get("decode_s", wall_seconds)),
        "ms_per_step": float(last_timing.get("ms_per_step", 0.0)),
    }
    return combined, sample_rate, timing


class PauseCut(NamedTuple):
    cut_sample: int
    pause_start_sample: int
    pause_end_sample: int


def find_pause_cut(
    audio: Any,
    sample_rate: int,
    *,
    expected_pause_start_sample: int,
    search_window_ms: int,
    min_pause_ms: int,
    lead_in_ms: int,
    threshold_db: float,
) -> PauseCut | None:
    """Find the pause nearest the calibrated preroll end using frame RMS."""
    import numpy as np

    array = _to_float32_audio(audio)
    if not len(array) or sample_rate <= 0:
        return None

    frame_samples = max(1, round(sample_rate * 0.020))
    hop_samples = max(1, round(sample_rate * 0.010))
    starts = np.arange(0, max(1, len(array) - frame_samples + 1), hop_samples)
    if not len(starts):
        return None
    frame_rms = np.asarray(
        [
            math.sqrt(
                float(
                    np.mean(
                        array[start : start + frame_samples].astype(np.float64)
                        ** 2
                    )
                )
            )
            for start in starts
        ],
        dtype=np.float64,
    )
    peak_rms = float(np.max(frame_rms)) if len(frame_rms) else 0.0
    absolute_threshold = 10 ** (threshold_db / 20)
    relative_threshold = peak_rms * 10 ** (-40.0 / 20)
    silence_threshold = max(absolute_threshold, relative_threshold)
    silent = frame_rms <= silence_threshold

    window_samples = round(search_window_ms * sample_rate / 1000)
    search_start = max(0, expected_pause_start_sample - window_samples)
    search_end = min(len(array), expected_pause_start_sample + window_samples)
    min_pause_samples = round(min_pause_ms * sample_rate / 1000)
    lead_in_samples = round(lead_in_ms * sample_rate / 1000)

    candidates: list[PauseCut] = []
    run_start: int | None = None
    for frame_index, is_silent in enumerate(silent.tolist() + [False]):
        if is_silent and run_start is None:
            run_start = frame_index
            continue
        if is_silent or run_start is None:
            continue

        run_end = frame_index
        pause_start = int(starts[run_start])
        last_silent_start = int(starts[run_end - 1])
        pause_end = min(len(array), last_silent_start + frame_samples)
        run_start = None
        if pause_end - pause_start < min_pause_samples:
            continue
        if not search_start <= pause_start <= search_end:
            continue
        if run_end >= len(silent):
            continue
        candidates.append(
            PauseCut(
                cut_sample=max(0, pause_end - lead_in_samples),
                pause_start_sample=pause_start,
                pause_end_sample=pause_end,
            )
        )

    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: abs(
            candidate.pause_start_sample - expected_pause_start_sample
        ),
    )


class TextPrerollChunkGenerator:
    """Wrap non-streaming chunk generation with calibrated text preroll."""

    def __init__(
        self,
        original_generator: Any,
        *,
        sentence: str,
        search_window_ms: int,
        min_pause_ms: int,
        lead_in_ms: int,
        debug_directory: Path,
        save_debug_wav: bool,
    ) -> None:
        sentence = sentence.strip()
        if not sentence:
            raise ValueError("text preroll sentence must not be empty")
        if sentence[-1] not in ".!?":
            sentence += "."
        self.original_generator = original_generator
        self.sentence = sentence
        self.search_window_ms = search_window_ms
        self.min_pause_ms = min_pause_ms
        self.lead_in_ms = lead_in_ms
        self.debug_directory = debug_directory
        self.save_debug_wav = save_debug_wav
        self.chunk_index = 0
        self.expected_pause_start_sample: int | None = None
        self.calibration_sample_rate: int | None = None

    def _calibrate(
        self,
        model: Any,
        voice_clone_prompt: Any,
        config: dict[str, Any],
        fast_generate: Any,
        *,
        run_warmup: bool,
        seed: int,
    ) -> tuple[dict[str, Any], bool]:
        calibration_config = {**config, "codec_context_frames": 0}
        calibration_audio, sample_rate, timing = self.original_generator(
            model,
            voice_clone_prompt,
            self.sentence,
            calibration_config,
            fast_generate,
            run_warmup=run_warmup,
            seed=seed,
            codec_context_state={"codes": None},
        )
        _, trailing = edge_silence_samples(
            calibration_audio, config["silence_threshold_db"]
        )
        spoken_end = max(1, len(calibration_audio) - trailing)
        self.expected_pause_start_sample = spoken_end
        self.calibration_sample_rate = sample_rate
        print(
            "Text preroll calibration: expected pause starts near "
            f"{spoken_end / sample_rate:.3f}s"
        )
        if self.save_debug_wav:
            self.debug_directory.mkdir(parents=True, exist_ok=True)
            write_wav(
                self.debug_directory / "text-preroll-calibration.wav",
                calibration_audio,
                sample_rate,
            )
        return timing, run_warmup

    def __call__(
        self,
        model: Any,
        voice_clone_prompt: Any,
        text: str,
        config: dict[str, Any],
        fast_generate: Any,
        *,
        run_warmup: bool,
        seed: int,
        codec_context_state: dict[str, Any] | None = None,
    ) -> tuple[Any, int, dict[str, Any]]:
        self.chunk_index += 1
        calibration_timing: dict[str, Any] | None = None
        calibration_used_warmup = False
        if self.expected_pause_start_sample is None:
            calibration_timing, calibration_used_warmup = self._calibrate(
                model,
                voice_clone_prompt,
                config,
                fast_generate,
                run_warmup=run_warmup,
                seed=seed,
            )

        combined_text = f"{self.sentence}\n\n{text.lstrip()}"
        audio, sample_rate, timing = self.original_generator(
            model,
            voice_clone_prompt,
            combined_text,
            config,
            fast_generate,
            run_warmup=run_warmup and not calibration_used_warmup,
            seed=seed,
            codec_context_state=codec_context_state,
        )
        if calibration_timing is not None:
            timing["tts_wall_s"] += calibration_timing["tts_wall_s"]
            timing["warmup_s"] += calibration_timing["warmup_s"]
        if sample_rate != self.calibration_sample_rate:
            raise RuntimeError(
                "sample rate changed between text preroll calibration and generation"
            )

        cut = find_pause_cut(
            audio,
            sample_rate,
            expected_pause_start_sample=self.expected_pause_start_sample,
            search_window_ms=self.search_window_ms,
            min_pause_ms=self.min_pause_ms,
            lead_in_ms=self.lead_in_ms,
            threshold_db=config["silence_threshold_db"],
        )
        if cut is None:
            self.debug_directory.mkdir(parents=True, exist_ok=True)
            failure_path = (
                self.debug_directory
                / f"text-preroll-cut-failed-chunk-{self.chunk_index:03d}.wav"
            )
            write_wav(failure_path, audio, sample_rate)
            raise RuntimeError(
                "no suitable text-preroll pause found for chunk "
                f"{self.chunk_index}; uncut audio saved to {failure_path}"
            )

        print(
            f"Text preroll: pause {cut.pause_start_sample / sample_rate:.3f}s-"
            f"{cut.pause_end_sample / sample_rate:.3f}s, cut at "
            f"{cut.cut_sample / sample_rate:.3f}s"
        )
        timing["text_preroll_cut_s"] = cut.cut_sample / sample_rate
        return audio[cut.cut_sample :], sample_rate, timing


def prepare_generation_text(text: str, config: dict[str, Any]) -> str:
    """Add the configured non-spoken tail marker to a TTS request."""
    prepared = text.rstrip()
    if not config["append_chunk_end_padding"]:
        return prepared
    marker = config["chunk_end_padding_text"]
    if prepared.endswith(marker):
        return prepared
    return f"{prepared}{marker}"


def generate_mp3(
    chunks: list[str] | list[TextChunk],
    output: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from faster_qwen3_tts import FasterQwen3TTS
    from faster_qwen3_tts.generate import fast_generate

    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    normalized_chunks = [
        chunk if isinstance(chunk, TextChunk) else TextChunk(chunk, "sentence")
        for chunk in chunks
    ]
    if normalized_chunks:
        normalized_chunks[-1] = TextChunk(normalized_chunks[-1].text, "end")

    total_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print(f"Loading model from {config['model_path']}...")
    model_start = time.perf_counter()
    model = FasterQwen3TTS.from_pretrained(
        config["model_path"],
        device=config["model_device"],
        dtype=dtype_by_name[config["dtype"]],
        max_seq_len=config["model_max_seq_len"],
    )
    # Predictor sampling is embedded in its CUDA graph, so set it before the
    # first capture. The existing defaults are unchanged (0.9 / top-k 50).
    model.predictor_graph.temperature = config["temperature"]
    model.predictor_graph.top_k = config["top_k"]
    model.predictor_graph.top_p = config["top_p"]
    model.predictor_graph.do_sample = config["do_sample"]
    model_init_seconds = time.perf_counter() - model_start

    prompt_label = (
        f"ICL prompt from {config['ref_audio']}"
        if config["mode"] == "semantic_icl"
        else f"speaker embedding from {config['speaker']}"
    )
    print(f"Loading {prompt_label}...")
    prompt_start = time.perf_counter()
    voice_clone_prompt = load_voice_prompt(model, config)
    prompt_init_seconds = time.perf_counter() - prompt_start

    parts_directory = output.with_name(f"{output.stem}_parts")
    if config["save_wav_parts"]:
        parts_directory.mkdir(parents=True, exist_ok=True)
    chunk_generator = _generate_audio_chunk
    preroll_generator: TextPrerollChunkGenerator | None = None
    if config["text_preroll_enabled"]:
        if config["generation_api"] != "non_streaming":
            raise ValueError(
                "text preroll requires generation_api 'non_streaming'"
            )
        preroll_generator = TextPrerollChunkGenerator(
            chunk_generator,
            sentence=config["text_preroll_sentence"],
            search_window_ms=config["text_preroll_search_window_ms"],
            min_pause_ms=config["text_preroll_min_pause_ms"],
            lead_in_ms=config["text_preroll_lead_in_ms"],
            debug_directory=parts_directory,
            save_debug_wav=config["save_wav_parts"],
        )
        chunk_generator = preroll_generator
        print(f"Fixed text preroll: {preroll_generator.sentence}")
    if config["append_chunk_end_padding"]:
        print("Chunk end padding enabled: two line breaks and a period")
    part_number_width = max(3, len(str(len(chunks))))
    audio_chunks: list[Any] = []
    sample_rate: int | None = None
    total_tts_seconds = 0.0
    total_warmup_seconds = 0.0
    postprocessing_seconds = 0.0
    chunk_metrics: list[dict[str, Any]] = []
    reference_rms: float | None = None
    codec_context_state: dict[str, Any] = {"codes": None}

    print(f"Processing {len(chunks)} text chunk(s)...", flush=True)
    if config["generation_api"] == "non_streaming":
        if config["codec_context_frames"] > 0:
            print(
                "Cross-chunk codec context enabled: "
                f"{config['codec_context_frames']} frame(s)"
            )
        else:
            print("Cross-chunk codec context disabled")
    elif config["codec_context_frames"] > 0:
        print(
            "Cross-chunk codec context is not applied with generation_api "
            "'streaming'; that API manages only its internal decode windows"
        )
    displayed_limit = (
        config["legacy_chunk_chars"]
        if config["mode"] == "legacy"
        else config["max_chunk_chars"]
    )
    for index, chunk in enumerate(normalized_chunks, start=1):
        print(
            f"\nChunk {index}/{len(chunks)}: "
            f"{len(chunk.text)}/{displayed_limit} characters "
            f"(boundary: {chunk.boundary_after})",
            flush=True,
        )
        seed = config["seed"] + (
            index - 1 if config["seed_strategy"] == "increment" else 0
        )
        generation_text = prepare_generation_text(chunk.text, config)
        if config["generation_api"] == "streaming":
            audio, chunk_sample_rate, timing = _generate_audio_chunk_streaming(
                model,
                voice_clone_prompt,
                generation_text,
                config,
                run_warmup=index == 1,
                seed=seed,
            )
        else:
            audio, chunk_sample_rate, timing = chunk_generator(
                model,
                voice_clone_prompt,
                generation_text,
                config,
                fast_generate,
                run_warmup=index == 1,
                seed=seed,
                codec_context_state=codec_context_state,
            )
        if sample_rate is None:
            sample_rate = chunk_sample_rate
        elif sample_rate != chunk_sample_rate:
            raise RuntimeError(
                f"sample rate changed between chunks: "
                f"{sample_rate} Hz vs {chunk_sample_rate} Hz"
            )

        post_start = time.perf_counter()
        audio = trim_excess_edge_silence(
            audio,
            chunk_sample_rate,
            threshold_db=config["silence_threshold_db"],
            max_leading_ms=config["max_leading_silence_ms"],
            max_trailing_ms=config["max_trailing_silence_ms"],
        )
        audio, observed_rms = match_chunk_loudness(
            audio,
            reference_rms,
            threshold_db=config["silence_threshold_db"],
            max_adjustment_db=config["loudness_match_max_db"],
        )
        if reference_rms is None and observed_rms > 0:
            reference_rms = observed_rms
        part_path = parts_directory / f"teil-{index:0{part_number_width}d}.wav"
        if config["save_wav_parts"]:
            write_wav(part_path, audio, chunk_sample_rate)
        postprocessing_seconds += time.perf_counter() - post_start
        audio_chunks.append(audio)

        steps = timing["steps"]
        audio_duration = len(audio) / chunk_sample_rate
        generation_time = timing["tts_wall_s"]
        total_tts_seconds += generation_time
        total_warmup_seconds += timing["warmup_s"]
        rtf = generation_time / audio_duration if audio_duration else 0.0
        chunk_metrics.append(
            {
                "index": index,
                "characters": len(chunk.text),
                "boundary_after": chunk.boundary_after,
                "seed": seed,
                "audio_seconds": audio_duration,
                "tts_seconds": generation_time,
                "rtf": rtf,
                "prefill_seconds": timing["prefill_ms"] / 1000,
                "decode_seconds": timing["decode_s"],
                "codec_decode_seconds": timing["codec_decode_s"],
                "codec_context_frames": int(
                    timing.get("codec_context_frames", 0)
                ),
                "steps": steps,
            }
        )
        print(
            f"{'Saved ' + str(part_path) if config['save_wav_parts'] else 'Generated'} "
            f"({audio_duration:.1f}s audio, {generation_time:.2f}s TTS, "
            f"RTF {rtf:.2f})"
        )
        print(
            f"  Prefill: {timing['prefill_ms']:.0f}ms | "
            f"Decode: {steps} steps @ {timing['ms_per_step']:.1f}ms/step"
        )

    if sample_rate is None:
        raise RuntimeError("generation produced no audio chunks")

    post_start = time.perf_counter()
    combined_audio = join_audio_chunks(
        audio_chunks, normalized_chunks, sample_rate, config
    )
    postprocessing_seconds += time.perf_counter() - post_start
    encode_start = time.perf_counter()
    write_mp3(output, combined_audio, sample_rate)
    mp3_encoding_seconds = time.perf_counter() - encode_start
    parts_deleted = False
    if (
        config["save_wav_parts"]
        and config["parts_directory_policy"] == "delete"
    ):
        delete_parts_directory(parts_directory, output)
        parts_deleted = True
    total_audio_duration = len(combined_audio) / sample_rate
    total_rtf = (
        total_tts_seconds / total_audio_duration if total_audio_duration else 0.0
    )
    peak_allocated = (
        torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0
    )
    total_seconds = time.perf_counter() - total_start
    report = {
        "model": config["model_path"],
        "dtype": config["dtype"],
        "mode": config["mode"],
        "prompt_mode": "icl" if config["mode"] == "semantic_icl" else "x_vector",
        "generation_api": config["generation_api"],
        "codec_context_frames": config["codec_context_frames"],
        "text_preroll_enabled": config["text_preroll_enabled"],
        "text_preroll_sentence": (
            preroll_generator.sentence if preroll_generator is not None else ""
        ),
        "append_chunk_end_padding": config["append_chunk_end_padding"],
        "chunk_end_padding_text": config["chunk_end_padding_text"],
        "output": str(output.resolve()),
        "total_characters": sum(len(chunk.text) for chunk in normalized_chunks),
        "chunk_count": len(normalized_chunks),
        "average_chunk_characters": (
            sum(len(chunk.text) for chunk in normalized_chunks) / len(normalized_chunks)
            if normalized_chunks
            else 0.0
        ),
        "audio_seconds": total_audio_duration,
        "total_seconds": total_seconds,
        "model_init_seconds": model_init_seconds,
        "prompt_init_seconds": prompt_init_seconds,
        "warmup_seconds": total_warmup_seconds,
        "tts_seconds": total_tts_seconds,
        "postprocessing_seconds": postprocessing_seconds,
        "mp3_encoding_seconds": mp3_encoding_seconds,
        "rtf": total_rtf,
        "x_realtime": total_audio_duration / total_tts_seconds if total_tts_seconds else 0.0,
        "peak_vram_allocated_gib": peak_allocated,
        "peak_vram_reserved_gib": peak_reserved,
        "chunks": chunk_metrics,
    }
    print(
        f"\nCombined {len(chunks)} lossless PCM part(s) into {output} "
        f"({total_audio_duration:.1f}s audio, "
        f"{total_tts_seconds:.2f}s TTS, RTF {total_rtf:.2f}, "
        f"{report['x_realtime']:.2f}x realtime)"
    )
    print(
        f"Init {model_init_seconds:.2f}s | warm-up {total_warmup_seconds:.2f}s | "
        f"post {postprocessing_seconds:.2f}s | MP3 {mp3_encoding_seconds:.2f}s | "
        f"peak VRAM {peak_allocated:.2f} GiB allocated"
    )
    if parts_deleted:
        print(f"Deleted temporary WAV parts: {parts_directory}")
    elif config["save_wav_parts"]:
        print(f"Lossless WAV parts: {parts_directory}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        text = resolve_text(args.text, args.input)
        output = resolve_output_path(args.output, args.input)
        config = load_config(
            args.config,
            cli_overrides={
                "speaker": args.speaker,
                "language": args.language,
                "model_path": args.model_path,
                "device": args.device,
                "dtype": args.dtype,
                "seed": args.seed,
                "mode": args.mode,
                "generation_api": args.generation_api,
                "target_chunk_chars": args.target_chars,
                "max_chunk_chars": args.characters,
                "ref_audio": (
                    str(args.ref_audio.resolve()) if args.ref_audio is not None else None
                ),
                "ref_text": args.ref_text,
                "ref_text_file": (
                    str(args.ref_text_file.resolve())
                    if args.ref_text_file is not None
                    else ("" if args.ref_text is not None else None)
                ),
                "instruct": args.instruct,
            },
        )
        if args.no_wav_parts:
            config["save_wav_parts"] = False
        if args.mode == "legacy" and args.characters is not None:
            config["legacy_chunk_chars"] = args.characters
        chunks = build_text_chunks(
            text,
            config,
            sentences_per_chunk=args.sentences,
            padding=args.padding,
        )
    except ValueError as exc:
        parser.error(str(exc))

    report = generate_mp3(chunks, output, config)
    if args.metrics is not None:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Metrics: {args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
