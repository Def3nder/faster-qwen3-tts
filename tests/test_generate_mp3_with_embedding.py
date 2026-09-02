import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import soundfile as sf


SCRIPT_PATH = Path(__file__).parents[1] / "generate" / "generate_mp3_with_embedding.py"
SPEC = importlib.util.spec_from_file_location("generate_mp3_with_embedding", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_config(path: Path, speaker: Path, **overrides):
    config = {"speaker": str(speaker), **overrides}
    path.write_text(json.dumps(config), encoding="utf-8")


class GenerateMp3WithEmbeddingTests(unittest.TestCase):
    def test_parser_requires_exactly_one_text_source(self):
        parser = MODULE.build_parser()

        direct = parser.parse_args(["--text", "Hallo", "--output", "out.mp3"])
        self.assertEqual(direct.text, "Hallo")
        self.assertIsNone(direct.input)

        from_file = parser.parse_args(["--input", "input.txt"])
        self.assertIsNone(from_file.text)
        self.assertEqual(from_file.input, Path("input.txt"))
        self.assertIsNone(from_file.output)

        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--text", "Hallo", "--input", "input.txt", "--output", "out.mp3"]
            )

    def test_parser_accepts_all_original_tts_arguments(self):
        parser = MODULE.build_parser()
        args = parser.parse_args(
            [
                "--speaker",
                "voice.pt",
                "--text",
                "Hallo",
                "--language",
                "German",
                "--output",
                "out.mp3",
                "--model_path",
                "local-model",
                "--device",
                "cuda:1",
            ]
        )

        self.assertEqual(args.speaker, "voice.pt")
        self.assertEqual(args.text, "Hallo")
        self.assertEqual(args.language, "German")
        self.assertEqual(args.output, Path("out.mp3"))
        self.assertEqual(args.model_path, "local-model")
        self.assertEqual(args.device, "cuda:1")

    def test_parser_accepts_positive_sentences_argument(self):
        parser = MODULE.build_parser()

        args = parser.parse_args(
            ["--text", "Hallo.", "--output", "out.mp3", "--sentences", "3"]
        )

        self.assertEqual(args.sentences, 3)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--text", "Hallo.", "--output", "out.mp3", "--sentences", "0"]
            )

    def test_parser_accepts_characters_within_supported_range(self):
        parser = MODULE.build_parser()

        args = parser.parse_args(
            ["--text", "Hallo.", "--output", "out.mp3", "--characters", "1500"]
        )

        self.assertEqual(args.characters, 1500)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--text",
                    "Hallo.",
                    "--output",
                    "out.mp3",
                    "--characters",
                    "3001",
                ]
            )

    def test_parser_accepts_padding_flag(self):
        parser = MODULE.build_parser()

        without_padding = parser.parse_args(
            ["--text", "Hallo.", "--output", "out.mp3"]
        )
        with_padding = parser.parse_args(
            ["--text", "Hallo.", "--output", "out.mp3", "--padding"]
        )

        self.assertFalse(without_padding.padding)
        self.assertTrue(with_padding.padding)

    def test_parser_accepts_clear_markdown_flag_and_alias(self):
        parser = MODULE.build_parser()

        plain = parser.parse_args(["--text", "Hallo.", "--output", "out.mp3"])
        underscored = parser.parse_args(
            ["--text", "Hallo.", "--output", "out.mp3", "--clear_markdown"]
        )
        dashed = parser.parse_args(
            ["--text", "Hallo.", "--output", "out.mp3", "--clear-markdown"]
        )
        disabled_underscored = parser.parse_args(
            ["--text", "Hallo.", "--output", "out.mp3", "--no_clear_markdown"]
        )
        disabled_dashed = parser.parse_args(
            ["--text", "Hallo.", "--output", "out.mp3", "--no-clear-markdown"]
        )

        self.assertIsNone(plain.clear_markdown)
        self.assertTrue(underscored.clear_markdown)
        self.assertTrue(dashed.clear_markdown)
        self.assertFalse(disabled_underscored.clear_markdown)
        self.assertFalse(disabled_dashed.clear_markdown)

    def test_parser_accepts_cleaned_text_output_switches_and_aliases(self):
        parser = MODULE.build_parser()

        underscored = parser.parse_args(
            [
                "--text",
                "Hallo.",
                "--clear_markdown",
                "--print_cleaned_text",
                "--write_cleaned_text",
                "cleaned.md",
            ]
        )
        dashed = parser.parse_args(
            [
                "--text",
                "Hallo.",
                "--clear-markdown",
                "--print-cleaned-text",
                "--write-cleaned-text",
                "cleaned.md",
            ]
        )

        self.assertTrue(underscored.print_cleaned_text)
        self.assertEqual(underscored.write_cleaned_text, Path("cleaned.md"))
        self.assertTrue(dashed.print_cleaned_text)
        self.assertEqual(dashed.write_cleaned_text, Path("cleaned.md"))

    def test_resolve_text_reads_utf8_bom_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.txt"
            input_path.write_text("\ufeff  Grüße aus Köln  \n", encoding="utf-8")

            self.assertEqual(
                MODULE.resolve_text(None, input_path),
                "Grüße aus Köln",
            )

    def test_load_config_merges_defaults_and_resolves_relative_speaker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            speaker = directory / "voice.pt"
            speaker.touch()
            config_path = directory / "config.json"
            _write_config(config_path, Path("voice.pt"), temperature=0.7)

            config = MODULE.load_config(config_path)

            self.assertEqual(config["speaker"], speaker)
            self.assertEqual(config["temperature"], 0.7)
            self.assertEqual(config["top_k"], 50)
            self.assertEqual(config["max_new_tokens"], 768)
            self.assertEqual(config["model_max_seq_len"], 2048)
            self.assertEqual(config["codec_context_frames"], 8)
            self.assertEqual(config["min_chunk_chars"], 220)
            self.assertEqual(config["target_chunk_chars"], 340)
            self.assertEqual(config["max_chunk_chars"], 520)
            self.assertTrue(config["text_preroll_enabled"])
            self.assertTrue(config["text_preroll_sentence"].endswith("\n\n."))
            self.assertTrue(config["append_chunk_end_padding"])
            self.assertEqual(config["chunk_end_padding_text"], "\n\n.")
            self.assertTrue(config["clear_markdown"])

    def test_resolve_clear_markdown_setting_uses_cli_then_config_then_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"

            self.assertTrue(
                MODULE.resolve_clear_markdown_setting(config_path, True)
            )
            self.assertFalse(
                MODULE.resolve_clear_markdown_setting(config_path, False)
            )

            config_path.write_text(
                json.dumps({"clear_markdown": False}),
                encoding="utf-8",
            )
            self.assertFalse(
                MODULE.resolve_clear_markdown_setting(config_path, None)
            )

            config_path.write_text(json.dumps({}), encoding="utf-8")
            self.assertTrue(
                MODULE.resolve_clear_markdown_setting(config_path, None)
            )

            config_path.write_text(
                json.dumps({"clear_markdown": "yes"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "clear_markdown.*must be bool"):
                MODULE.resolve_clear_markdown_setting(config_path, None)

    def test_cli_tts_arguments_override_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            config_speaker = directory / "config-voice.pt"
            config_speaker.touch()
            cli_speaker = directory / "cli-voice.pt"
            cli_speaker.touch()
            config_path = directory / "config.json"
            _write_config(
                config_path,
                config_speaker,
                language="English",
                model_path="config-model",
                device="cuda:0",
            )

            config = MODULE.load_config(
                config_path,
                {
                    "speaker": str(cli_speaker),
                    "language": "German",
                    "model_path": "cli-model",
                    "device": "cuda:1",
                    "max_chunk_chars": 1200,
                },
            )

            self.assertEqual(config["speaker"], cli_speaker)
            self.assertEqual(config["language"], "German")
            self.assertEqual(config["model_path"], "cli-model")
            self.assertEqual(config["device"], "cuda:1")
            self.assertEqual(config["max_chunk_chars"], 1200)

    def test_relative_cli_speaker_is_resolved_from_current_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            cli_speaker = directory / "cli-voice.pt"
            cli_speaker.touch()
            config_path = directory / "config.json"
            _write_config(config_path, Path("missing-config-voice.pt"))
            previous_directory = Path.cwd()

            try:
                os.chdir(directory)
                config = MODULE.load_config(
                    config_path,
                    {"speaker": "cli-voice.pt"},
                )
            finally:
                os.chdir(previous_directory)

            self.assertEqual(config["speaker"], cli_speaker)

    def test_load_config_rejects_unknown_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            speaker = directory / "voice.pt"
            speaker.touch()
            config_path = directory / "config.json"
            _write_config(config_path, speaker, typo_value=True)

            with self.assertRaisesRegex(ValueError, "unknown config value"):
                MODULE.load_config(config_path)

    def test_load_config_rejects_chunk_limit_above_3000(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            speaker = directory / "voice.pt"
            speaker.touch()
            config_path = directory / "config.json"
            _write_config(config_path, speaker, max_chunk_chars=3001)

            with self.assertRaisesRegex(ValueError, "between 1 and 3000"):
                MODULE.load_config(config_path)

    def test_parser_accepts_generation_and_reference_arguments(self):
        parser = MODULE.build_parser()

        args = parser.parse_args(
            [
                "--text",
                "Hallo.",
                "--output",
                "out.mp3",
                "--dtype",
                "float16",
                "--seed",
                "42",
                "--mode",
                "semantic_icl",
                "--generation-api",
                "non_streaming",
                "--target-chars",
                "300",
                "--ref-audio",
                "reference.wav",
                "--ref-text-file",
                "reference.txt",
                "--instruct",
                "Ruhig sprechen",
                "--metrics",
                "metrics.json",
                "--no-wav-parts",
            ]
        )

        self.assertEqual(args.dtype, "float16")
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.mode, "semantic_icl")
        self.assertEqual(args.generation_api, "non_streaming")
        self.assertEqual(args.target_chars, 300)
        self.assertEqual(args.ref_audio, Path("reference.wav"))
        self.assertEqual(args.ref_text_file, Path("reference.txt"))
        self.assertEqual(args.instruct, "Ruhig sprechen")
        self.assertEqual(args.metrics, Path("metrics.json"))
        self.assertTrue(args.no_wav_parts)

    def test_parser_rejects_missing_text_source(self):
        parser = MODULE.build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["--output", "out.mp3"])

    def test_resolve_text_supports_direct_text_and_reports_input_errors(self):
        self.assertEqual(MODULE.resolve_text("  Direkt eingegeben  ", None), "Direkt eingegeben")

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            MODULE.resolve_text("   ", None)

        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            missing = directory / "missing.txt"
            with self.assertRaisesRegex(ValueError, "not found"):
                MODULE.resolve_text(None, missing)

            invalid_utf8 = directory / "invalid.txt"
            invalid_utf8.write_bytes(b"\xff\xfe\xfa")
            with self.assertRaisesRegex(ValueError, "not valid UTF-8"):
                MODULE.resolve_text(None, invalid_utf8)

    def test_clear_markdown_normalizes_abbreviations_and_list_markers(self):
        text = (
            "Das ist z. B. wichtig. Das ist z.\u00a0B. neu.\n"
            "Das ist d. h. klar. Das ist d.\u00a0h. sicher.\n"
            "Wir nutzen u. a. Äpfel, u.a. Birnen, ggf. Kirschen bzw. Pflaumen.\n"
            "+ Erster Punkt\n"
            "Der Ablauf: 1. starten, 2. prüfen und 21. abschließen.\n"
            "1. Listenpunkt am Zeilenanfang."
        )

        self.assertEqual(
            MODULE.clear_markdown_text(text),
            "Das ist z.B. wichtig. Das ist z.B. neu.\n"
            "Das ist d.h. klar. Das ist d.h. sicher.\n"
            "Wir nutzen unter anderem Äpfel, unter anderem Birnen, "
            "gegebenenfalls Kirschen beziehungsweise Pflaumen.\n"
            "- Erster Punkt\n"
            "Der Ablauf: erstens starten, zweitens prüfen und "
            "einundzwanzigstens abschließen.\n"
            "1. Listenpunkt am Zeilenanfang.",
        )

    def test_clear_markdown_writes_valid_dates_in_german_long_form(self):
        text = (
            "01.08.2026\n"
            "2026-08-01\n"
            "02.08.2026\n"
            "2026-08-03\n"
            "29.02.2024\n"
            "2024-02-29\n"
            "31.02.2026\n"
            "Version 2.5.1"
        )

        self.assertEqual(
            MODULE.clear_markdown_text(text),
            "Erster August 2026\n"
            "Erster August 2026\n"
            "Zweiter August 2026\n"
            "Dritter August 2026\n"
            "Neunundzwanzigster Februar 2024\n"
            "Neunundzwanzigster Februar 2024\n"
            "31.02.2026\n"
            "Version 2.5.1",
        )

    def test_clear_markdown_removes_fenced_code_and_urls_but_keeps_inline_code(self):
        text = (
            "Vorher.\n\n"
            "```python\n"
            "print('nicht sprechen')\n"
            "```\n\n"
            "Nachher mit `inline_code` und ```inline fenced text```.\n"
            "Quelle: http://example.org/quelle\n"
            "Quelle: https://example.org/quelle\n"
            "Besuche https://example.org/test.\n"
            "Siehe [Beispiel](https://example.org/page).\n"
            "Auch www.example.org ist eine URL.\n"
            "E-Mail test@example.org bleibt."
        )

        self.assertEqual(
            MODULE.clear_markdown_text(text),
            "Vorher.\n\n"
            "Nachher mit `inline_code` und ```inline fenced text```.\n"
            "Besuche.\n"
            "Siehe Beispiel.\n"
            "Auch ist eine URL.\n"
            "E-Mail test@example.org bleibt.",
        )

    def test_clear_markdown_removes_date_lines_without_calendar_validation(self):
        text = (
            "Vorher.\n"
            "Datum: 01.08.2026\n"
            "  datum: 2026-08-02.  \n"
            "Quelle: 03.08.2026\n"
            "Datum: 31.02.2026\n"
            "Datum: 99.99.9999\n"
            "Datum: 01.08.2026, Seite 4\n"
            "Quelle: https://example.org/beleg\n"
            "Nachher."
        )

        self.assertEqual(
            MODULE.clear_markdown_text(text),
            "Vorher.\n"
            "Quelle: Dritter August 2026\n"
            "Datum: Erster August 2026, Seite 4\n"
            "Nachher.",
        )

    def test_clear_markdown_removes_markdown_formatted_source_and_date_lines(self):
        text = (
            "Vorher.\n"
            "_Quelle: https://sample.com/quelle_\n"
            "**Quelle: https:://sample.com/quelle**\n"
            "*Quelle: https://sample.com/quelle*\n"
            "**Quelle:** https://sample.com/quelle\n"
            "__Quelle__: https://sample.com/quelle\n"
            "> **Quelle:** https://sample.com/quelle\n"
            "_Datum: 01.08.2026_\n"
            "**Datum: 2026-08-01**\n"
            "*Datum: 31.02.2026*\n"
            "**Datum:** 99.99.9999\n"
            "__Datum__: **02.08.2026**\n"
            "### **Datum:** 2026-08-03\n"
            "Nachher."
        )

        self.assertEqual(
            MODULE.clear_markdown_text(text),
            "Vorher.\nNachher.",
        )

    def test_clear_markdown_preserves_unclosed_fence(self):
        text = "Vorher.\n```python\nprint('unvollständig')"

        self.assertEqual(MODULE.clear_markdown_text(text), text)

    def test_load_config_reports_file_and_json_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            missing = directory / "missing.json"
            with self.assertRaisesRegex(ValueError, "config file not found"):
                MODULE.load_config(missing)

            invalid = directory / "invalid.json"
            invalid.write_text("{not-json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                MODULE.load_config(invalid)

            non_object = directory / "array.json"
            non_object.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                MODULE.load_config(non_object)

    def test_load_config_rejects_invalid_types_values_and_dependencies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            speaker = directory / "voice.pt"
            speaker.touch()
            cases = [
                ({"temperature": "hot"}, "must be float"),
                ({"top_k": True}, "must be int"),
                ({"do_sample": 1}, "must be bool"),
                ({"clear_markdown": "yes"}, "clear_markdown.*must be bool"),
                ({"dtype": "int8"}, "dtype.*one of"),
                ({"mode": "unknown"}, "mode.*one of"),
                ({"generation_api": "batch"}, "generation_api.*one of"),
                ({"seed_strategy": "random"}, "seed_strategy.*one of"),
                ({"parts_directory_policy": "archive"}, "parts_directory_policy.*one of"),
                ({"temperature": -0.1}, "temperature.*>= 0"),
                ({"top_k": -1}, "top_k.*>= 0"),
                ({"top_p": 0.0}, r"top_p.*\(0, 1]"),
                ({"repetition_penalty": 0.0}, "repetition_penalty.*> 0"),
                ({"min_new_tokens": -1}, "token limits"),
                ({"warmup_max_new_tokens": 0}, "token limits"),
                ({"max_new_tokens": 0}, "token limits"),
                ({"model_max_seq_len": 0}, "token limits"),
                ({"model_max_seq_len": 768}, "greater than.*max_new_tokens"),
                ({"codec_context_frames": -1}, "codec_context_frames.*>= 0"),
                ({"legacy_chunk_chars": 0}, "chunk sizes"),
                ({"min_chunk_chars": 400, "target_chunk_chars": 300}, "min_chunk_chars"),
                ({"sentence_pause_ms": -1}, "pause, fade, trim"),
                ({"loudness_match_max_db": -0.1}, "loudness_match_max_db.*>= 0"),
                ({"generation_api": "streaming"}, "text preroll requires"),
                ({"text_preroll_sentence": " "}, "text_preroll_sentence.*empty"),
                ({"text_preroll_min_pause_ms": 0}, "text_preroll_min_pause_ms.*> 0"),
                ({"text_preroll_search_window_ms": -1}, "search window and lead-in"),
                ({"text_preroll_lead_in_ms": -1}, "search window and lead-in"),
                ({"chunk_end_padding_text": "."}, "must be exactly"),
            ]

            for index, (overrides, message) in enumerate(cases):
                with self.subTest(overrides=overrides):
                    config_path = directory / f"invalid-{index}.json"
                    _write_config(config_path, speaker, **overrides)
                    with self.assertRaisesRegex(ValueError, message):
                        MODULE.load_config(config_path)

    def test_load_config_resolves_speaker_fallback_and_icl_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            speaker = directory / "narrator-1.7B.pt"
            speaker.touch()
            ref_audio = directory / "reference.wav"
            ref_audio.touch()
            ref_text_file = directory / "reference.txt"
            ref_text_file.write_text("\ufeff  Exakter Referenztext.  ", encoding="utf-8")
            config_path = directory / "config.json"
            _write_config(
                config_path,
                Path("narrator"),
                mode="semantic_icl",
                ref_audio="reference.wav",
                ref_text_file="reference.txt",
            )

            config = MODULE.load_config(config_path)

            self.assertEqual(config["speaker"], speaker)
            self.assertEqual(config["ref_audio"], ref_audio)
            self.assertEqual(config["ref_text_file"], ref_text_file)
            self.assertEqual(config["ref_text"], "Exakter Referenztext.")

    def test_load_config_reports_missing_speaker_reference_and_icl_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            speaker = directory / "voice.pt"
            speaker.touch()

            missing_speaker_config = directory / "missing-speaker.json"
            _write_config(missing_speaker_config, Path("missing.pt"))
            with self.assertRaisesRegex(ValueError, "speaker embedding not found"):
                MODULE.load_config(missing_speaker_config)

            missing_reference_config = directory / "missing-reference.json"
            _write_config(
                missing_reference_config,
                speaker,
                ref_audio="missing.wav",
            )
            with self.assertRaisesRegex(ValueError, "does not exist"):
                MODULE.load_config(missing_reference_config)

            ref_audio = directory / "reference.wav"
            ref_audio.touch()
            missing_text_config = directory / "missing-text.json"
            _write_config(
                missing_text_config,
                speaker,
                mode="semantic_icl",
                ref_audio=str(ref_audio),
            )
            with self.assertRaisesRegex(ValueError, "requires both ref_audio"):
                MODULE.load_config(missing_text_config)

    def test_chunk_text_keeps_complete_paragraphs(self):
        first = "Erster Absatz."
        second = "Der zweite Absatz bleibt ebenfalls vollständig."
        third = "Dritter Absatz."
        max_chars = len(first) + 2 + len(second)

        chunks = MODULE.chunk_text(
            f"{first}\n\n{second}\n\n{third}",
            max_chars,
        )

        self.assertEqual(chunks, [f"{first}\n\n{second}", third])

    def test_oversized_paragraph_is_split_only_between_sentences(self):
        first = "Der erste Satz bleibt vollständig."
        second = "Auch der zweite Satz wird nicht zerteilt!"
        third = "Ist der dritte Satz ebenfalls vollständig? Ja."
        paragraph = f"{first} {second} {third}"

        chunks = MODULE.chunk_text(paragraph, max_chars=55)

        self.assertEqual(chunks, [first, second, third])
        self.assertTrue(all(len(chunk) <= 55 for chunk in chunks))

    def test_sentence_splitter_does_not_split_common_abbreviations(self):
        paragraph = "Dr. Müller nennt z.B. ein Beispiel. Danach geht es weiter."

        self.assertEqual(
            MODULE.split_sentences(paragraph),
            [
                "Dr. Müller nennt z.B. ein Beispiel.",
                "Danach geht es weiter.",
            ],
        )

    def test_sentence_splitter_does_not_split_numbered_list_markers(self):
        paragraph = "1. Der erste Punkt ist vollständig.\n2. Danach folgt Nummer zwei."

        self.assertEqual(
            MODULE.split_sentences(paragraph),
            [
                "1. Der erste Punkt ist vollständig.",
                "2. Danach folgt Nummer zwei.",
            ],
        )

    def test_german_numbered_lists_are_rewritten_for_speech(self):
        text = (
            "1. Vorbereitung\n"
            "2. Durchführung\n"
            "3. Abschluss\n"
            "21. Ausblick\n\n"
            "Am 1. Mai bleibt die Schreibweise erhalten. Version 1.5 ebenfalls."
        )

        self.assertEqual(
            MODULE.normalize_numbered_list_markers(text, "German"),
            "Erstens, Vorbereitung\n"
            "Zweitens, Durchführung\n"
            "Drittens, Abschluss\n"
            "Einundzwanzigstens, Ausblick\n\n"
            "Am 1. Mai bleibt die Schreibweise erhalten. Version 1.5 ebenfalls.",
        )

    def test_numbered_list_rewriting_is_german_only_and_can_be_disabled(self):
        text = "1. First item.\n2. Second item."
        self.assertEqual(MODULE.normalize_numbered_list_markers(text, "English"), text)

        config = {
            **MODULE.DEFAULT_CONFIG,
            "language": "German",
            "speak_numbered_lists": False,
        }
        chunks = MODULE.build_text_chunks(text, config)
        self.assertTrue(chunks[0].text.startswith("1. First item."))

    def test_chunk_text_accepts_exact_3000_character_boundary(self):
        text = f"{'A' * 2999}."

        self.assertEqual(MODULE.chunk_text(text), [text])

    def test_chunk_text_rejects_single_sentence_over_limit(self):
        sentence = f"{'A' * 3000}."

        with self.assertRaisesRegex(ValueError, "single sentence has 3001"):
            MODULE.chunk_text(sentence)

    def test_semantic_chunking_prefers_paragraph_and_topic_boundaries(self):
        paragraph = " ".join(
            f"Satz {index} beschreibt einen zusammenhängenden Gedanken ausführlich."
            for index in range(1, 9)
        )
        text = f"# Einleitung\n\n{paragraph}\n\n# Neues Thema\n\n{paragraph}"

        chunks = MODULE.semantic_chunk_text(
            text, target_chars=430, min_chars=300, max_chars=520
        )

        self.assertTrue(all(len(chunk.text) <= 520 for chunk in chunks))
        self.assertFalse(any("#" in chunk.text for chunk in chunks))
        self.assertIn("topic", [chunk.boundary_after for chunk in chunks])

    def test_semantic_chunking_splits_unavoidable_long_sentence_at_clause(self):
        text = "Dies ist ein langer Gedanke, " + "sehr ausführlich, " * 30 + "am Ende."

        chunks = MODULE.semantic_chunk_text(
            text, target_chars=120, min_chars=80, max_chars=150
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 150 for chunk in chunks))

    def test_build_text_chunks_keeps_legacy_comparison_mode(self):
        text = " ".join(f"Das ist Satz Nummer {index}." for index in range(40))
        semantic_config = {
            **MODULE.DEFAULT_CONFIG,
            "mode": "semantic",
            "min_chunk_chars": 400,
            "target_chunk_chars": 700,
            "max_chunk_chars": 900,
        }
        legacy_config = {**MODULE.DEFAULT_CONFIG, "mode": "legacy"}

        semantic = MODULE.build_text_chunks(text, semantic_config)
        legacy = MODULE.build_text_chunks(text, legacy_config)

        self.assertLess(len(semantic), len(legacy))
        self.assertTrue(all(len(chunk.text) <= 400 for chunk in legacy))

    def test_chunk_text_can_group_exact_sentence_counts(self):
        text = "Satz eins. Satz zwei!\n\nSatz drei? Satz vier. Satz fünf."

        chunks = MODULE.chunk_text(
            text,
            max_chars=100,
            sentences_per_chunk=2,
        )

        self.assertEqual(
            chunks,
            [
                "Satz eins. Satz zwei!",
                "Satz drei? Satz vier.",
                "Satz fünf.",
            ],
        )

    def test_sentence_count_mode_preserves_paragraph_boundary_in_chunk(self):
        text = "Satz eins. Satz zwei.\n\nSatz drei."

        self.assertEqual(
            MODULE.chunk_text(text, max_chars=100, sentences_per_chunk=3),
            ["Satz eins. Satz zwei.\n\nSatz drei."],
        )

    def test_sentence_count_mode_rejects_chunk_over_character_limit(self):
        text = "Dieser erste Satz ist länger. Dieser zweite Satz ist auch länger."

        with self.assertRaisesRegex(ValueError, "reduce --sentences"):
            MODULE.chunk_text(
                text,
                max_chars=40,
                sentences_per_chunk=2,
            )

    def test_sentence_count_mode_rejects_single_sentence_over_limit(self):
        sentence = f"{'A' * 100}."

        with self.assertRaisesRegex(ValueError, "single sentence has 101"):
            MODULE.chunk_text(
                sentence,
                max_chars=100,
                sentences_per_chunk=1,
            )

    def test_padding_appends_two_line_breaks_and_period_to_each_sentence(self):
        chunks = MODULE.chunk_text(
            "Satz eins. Satz zwei!",
            max_chars=100,
            padding=True,
        )

        self.assertEqual(chunks, ["Satz eins.\n\n. Satz zwei!\n\n."])

    def test_padding_works_with_sentence_count_mode(self):
        chunks = MODULE.chunk_text(
            "Satz eins. Satz zwei! Satz drei?",
            max_chars=100,
            sentences_per_chunk=2,
            padding=True,
        )

        self.assertEqual(
            chunks,
            [
                "Satz eins.\n\n. Satz zwei!\n\n.",
                "Satz drei?\n\n.",
            ],
        )

    def test_padding_is_included_in_character_limit(self):
        chunks = MODULE.chunk_text(
            "Eins. Zwei.",
            max_chars=10,
            padding=True,
        )

        self.assertEqual(chunks, ["Eins.\n\n.", "Zwei.\n\n."])
        self.assertTrue(all(len(chunk) <= 10 for chunk in chunks))

    def test_validate_output_path_requires_mp3(self):
        self.assertEqual(
            MODULE.validate_output_path(Path("out.MP3")),
            Path("out.MP3"),
        )

        with self.assertRaisesRegex(ValueError, r"\.mp3"):
            MODULE.validate_output_path(Path("out.wav"))

    def test_resolve_output_path_derives_mp3_from_input(self):
        self.assertEqual(
            MODULE.resolve_output_path(None, Path("texts") / "chapter.txt"),
            Path("texts") / "chapter.mp3",
        )
        self.assertEqual(
            MODULE.resolve_output_path(Path("audio.mp3"), Path("input.txt")),
            Path("audio.mp3"),
        )

    def test_resolve_output_path_requires_output_with_direct_text(self):
        with self.assertRaisesRegex(ValueError, "--output is required"):
            MODULE.resolve_output_path(None, None)

    def test_write_mp3_creates_mpeg_layer_iii_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "nested" / "silence.mp3"
            audio = np.zeros(2400, dtype=np.float32)

            MODULE.write_mp3(output, audio, 24000)

            info = sf.info(output)
            self.assertTrue(output.is_file())
            self.assertEqual(info.format, "MP3")
            self.assertEqual(info.subtype, "MPEG_LAYER_III")

    def test_join_audio_chunks_adds_only_missing_boundary_silence(self):
        config = {**MODULE.DEFAULT_CONFIG, "edge_fade_ms": 0}
        first = np.concatenate(
            [np.full(100, 0.2, dtype=np.float32), np.zeros(100, dtype=np.float32)]
        )
        second = np.concatenate(
            [np.zeros(50, dtype=np.float32), np.full(100, 0.2, dtype=np.float32)]
        )

        joined = MODULE.join_audio_chunks(
            [first, second],
            [MODULE.TextChunk("eins", "sentence"), MODULE.TextChunk("zwei", "end")],
            1000,
            config,
        )

        # Existing 150 ms edge silence plus 70 ms inserted reaches the 220 ms target.
        self.assertEqual(len(joined), len(first) + len(second) + 70)

    def test_generate_audio_chunk_carries_configured_codec_context(self):
        import torch

        class FakeTokenizer:
            def __init__(self):
                self.inputs = []

            def decode(self, payload):
                codes = payload["audio_codes"].squeeze(0).clone()
                self.inputs.append(codes)
                return [np.arange(codes.shape[0] * 10, dtype=np.float32)], 24000

        tokenizer = FakeTokenizer()
        prepared_model = mock.Mock()
        prepared_model.speech_tokenizer = tokenizer
        model = mock.Mock()
        model._prepare_generation.return_value = (
            prepared_model,
            mock.Mock(),
            mock.Mock(),
            torch.zeros((1, 3, 2)),
            mock.Mock(),
            mock.Mock(),
            mock.Mock(),
            None,
        )
        first_codes = torch.tensor([[1], [2], [3], [4]])
        second_codes = torch.tensor([[5], [6], [7]])
        timing = {
            "steps": 1,
            "prefill_ms": 1.0,
            "decode_s": 0.01,
            "ms_per_step": 10.0,
        }
        fast_generate = mock.Mock(
            side_effect=[(first_codes, dict(timing)), (second_codes, dict(timing))]
        )
        config = {
            **MODULE.DEFAULT_CONFIG,
            "speaker": Path("voice.pt"),
            "codec_context_frames": 2,
        }
        context_state = {"codes": None}

        first_audio, _, first_timing = MODULE._generate_audio_chunk(
            model,
            {},
            "Erster Satz.",
            config,
            fast_generate,
            run_warmup=False,
            seed=1,
            codec_context_state=context_state,
        )
        second_audio, _, second_timing = MODULE._generate_audio_chunk(
            model,
            {},
            "Zweiter Satz.",
            config,
            fast_generate,
            run_warmup=False,
            seed=2,
            codec_context_state=context_state,
        )

        self.assertEqual(tokenizer.inputs[0].flatten().tolist(), [1, 2, 3, 4])
        self.assertEqual(
            tokenizer.inputs[1].flatten().tolist(), [3, 4, 5, 6, 7]
        )
        self.assertEqual(len(first_audio), 40)
        self.assertEqual(len(second_audio), 30)
        self.assertEqual(first_timing["codec_context_frames"], 0)
        self.assertEqual(second_timing["codec_context_frames"], 2)

    def test_generate_mp3_writes_parts_and_combined_target(self):
        import faster_qwen3_tts

        timing = {
            "steps": 1,
            "prefill_ms": 10.0,
            "decode_s": 0.01,
            "ms_per_step": 10.0,
            "codec_decode_s": 0.01,
            "warmup_s": 0.02,
            "tts_wall_s": 0.03,
        }
        audio_parts = [
            np.full(2400, 0.1, dtype=np.float32),
            np.full(3600, -0.1, dtype=np.float32),
        ]
        generated = [
            (audio_parts[0], 24000, timing),
            (audio_parts[1], 24000, timing),
        ]
        config = {
            **MODULE.DEFAULT_CONFIG,
            "speaker": Path("voice.pt"),
            "parts_directory_policy": "keep",
            "model_max_seq_len": 16384,
            "text_preroll_enabled": False,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "result.mp3"
            fake_model = mock.Mock()
            with (
                mock.patch.object(
                    faster_qwen3_tts.FasterQwen3TTS,
                    "from_pretrained",
                    return_value=fake_model,
                ) as from_pretrained,
                mock.patch.object(MODULE, "load_voice_prompt", return_value={}),
                mock.patch.object(
                    MODULE,
                    "_generate_audio_chunk",
                    side_effect=generated,
                ) as generate_chunk,
                mock.patch("builtins.print") as print_mock,
            ):
                report = MODULE.generate_mp3(
                    ["Erster Satz.", "Zweiter Satz."],
                    output,
                    config,
                )

            parts_directory = Path(temp_dir) / "result_parts"
            first_part = parts_directory / "teil-001.wav"
            second_part = parts_directory / "teil-002.wav"
            self.assertTrue(first_part.is_file())
            self.assertTrue(second_part.is_file())
            self.assertTrue(output.is_file())
            self.assertEqual(
                from_pretrained.call_args.kwargs["max_seq_len"], 16384
            )
            self.assertGreater(sf.info(output).frames, sf.info(second_part).frames)
            self.assertTrue(generate_chunk.call_args_list[0].kwargs["run_warmup"])
            self.assertFalse(generate_chunk.call_args_list[1].kwargs["run_warmup"])
            first_context_state = generate_chunk.call_args_list[0].kwargs[
                "codec_context_state"
            ]
            second_context_state = generate_chunk.call_args_list[1].kwargs[
                "codec_context_state"
            ]
            self.assertIs(first_context_state, second_context_state)
            chunk_status_calls = [
                call
                for call in print_mock.call_args_list
                if call.args and str(call.args[0]).startswith("\nChunk ")
            ]
            self.assertEqual(
                [call.args[0] for call in chunk_status_calls],
                [
                    "\nChunk 1/2: 12/520 characters (boundary: sentence)",
                    "\nChunk 2/2: 13/520 characters (boundary: end)",
                ],
            )
            self.assertEqual(
                generate_chunk.call_args_list[0].args[2], "Erster Satz.\n\n."
            )
            self.assertEqual(
                generate_chunk.call_args_list[1].args[2], "Zweiter Satz.\n\n."
            )
            self.assertTrue(all(call.kwargs.get("flush") for call in chunk_status_calls))
            self.assertEqual(report["chunk_count"], 2)
            self.assertEqual(report["codec_context_frames"], 8)
            self.assertTrue(report["clear_markdown"])
            self.assertIn("peak_vram_allocated_gib", report)

    def test_generate_mp3_applies_preroll_and_end_padding_to_first_chunk(self):
        import faster_qwen3_tts

        timing = {
            "steps": 10,
            "prefill_ms": 10.0,
            "decode_s": 0.1,
            "ms_per_step": 10.0,
            "codec_decode_s": 0.01,
            "warmup_s": 0.02,
            "tts_wall_s": 0.12,
        }
        calibration_audio = np.concatenate(
            [
                np.full(24000, 0.2, dtype=np.float32),
                np.zeros(7200, dtype=np.float32),
            ]
        )
        combined_audio = np.concatenate(
            [
                np.full(24000, 0.2, dtype=np.float32),
                np.zeros(7200, dtype=np.float32),
                np.full(24000, 0.1, dtype=np.float32),
            ]
        )
        config = {
            **MODULE.DEFAULT_CONFIG,
            "speaker": Path("voice.pt"),
            "save_wav_parts": False,
            "text_preroll_sentence": "Ein kurzer Vorlauf.",
            "text_preroll_search_window_ms": 500,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "result.mp3"
            fake_model = mock.Mock()
            with (
                mock.patch.object(
                    faster_qwen3_tts.FasterQwen3TTS,
                    "from_pretrained",
                    return_value=fake_model,
                ),
                mock.patch.object(MODULE, "load_voice_prompt", return_value={}),
                mock.patch.object(
                    MODULE,
                    "_generate_audio_chunk",
                    side_effect=[
                        (calibration_audio, 24000, dict(timing)),
                        (combined_audio, 24000, dict(timing)),
                    ],
                ) as generate_chunk,
                mock.patch("builtins.print"),
            ):
                report = MODULE.generate_mp3(["Der Zieltext."], output, config)

        self.assertEqual(generate_chunk.call_count, 2)
        self.assertEqual(generate_chunk.call_args_list[0].args[2], "Ein kurzer Vorlauf.")
        self.assertEqual(
            generate_chunk.call_args_list[1].args[2],
            "Ein kurzer Vorlauf.\n\nDer Zieltext.\n\n.",
        )
        self.assertTrue(generate_chunk.call_args_list[0].kwargs["run_warmup"])
        self.assertFalse(generate_chunk.call_args_list[1].kwargs["run_warmup"])
        self.assertTrue(report["text_preroll_enabled"])
        self.assertTrue(report["append_chunk_end_padding"])
        self.assertEqual(report["chunk_end_padding_text"], "\n\n.")

    def test_generate_mp3_deletes_parts_after_success_by_default(self):
        import faster_qwen3_tts

        timing = {
            "steps": 1,
            "prefill_ms": 10.0,
            "decode_s": 0.01,
            "ms_per_step": 10.0,
            "codec_decode_s": 0.01,
            "warmup_s": 0.02,
            "tts_wall_s": 0.03,
        }
        config = {
            **MODULE.DEFAULT_CONFIG,
            "speaker": Path("voice.pt"),
            "text_preroll_enabled": False,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "result.mp3"
            parts_directory = Path(temp_dir) / "result_parts"
            fake_model = mock.Mock()
            with (
                mock.patch.object(
                    faster_qwen3_tts.FasterQwen3TTS,
                    "from_pretrained",
                    return_value=fake_model,
                ),
                mock.patch.object(MODULE, "load_voice_prompt", return_value={}),
                mock.patch.object(
                    MODULE,
                    "_generate_audio_chunk",
                    return_value=(
                        np.full(2400, 0.1, dtype=np.float32),
                        24000,
                        timing,
                    ),
                ),
                mock.patch("builtins.print"),
            ):
                MODULE.generate_mp3(["Ein Satz."], output, config)

            self.assertTrue(output.is_file())
            self.assertFalse(parts_directory.exists())

    def test_load_xvector_prompt_accepts_valid_tensor_and_rejects_wrong_size(self):
        import torch

        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            valid_path = directory / "valid.pt"
            valid_embedding = torch.arange(2048, dtype=torch.float32)
            torch.save(valid_embedding, valid_path)

            prompt = MODULE.load_xvector_prompt(valid_path, "cpu")

            self.assertTrue(torch.equal(prompt["ref_spk_embedding"][0], valid_embedding))
            self.assertEqual(prompt["ref_code"], [None])
            self.assertEqual(prompt["x_vector_only_mode"], [True])
            self.assertEqual(prompt["icl_mode"], [False])

            invalid_path = directory / "invalid.pt"
            torch.save(torch.zeros(16), invalid_path)
            with self.assertRaisesRegex(ValueError, "2048-element"):
                MODULE.load_xvector_prompt(invalid_path, "cpu")

    def test_load_voice_prompt_selects_xvector_loader(self):
        sentinel = object()
        config = {
            **MODULE.DEFAULT_CONFIG,
            "mode": "semantic",
            "speaker": Path("voice.pt"),
            "device": "cpu",
        }

        with mock.patch.object(
            MODULE, "load_xvector_prompt", return_value=sentinel
        ) as loader:
            result = MODULE.load_voice_prompt(mock.Mock(), config)

        self.assertIs(result, sentinel)
        loader.assert_called_once_with(Path("voice.pt"), device="cpu")

    def test_load_voice_prompt_builds_mono_icl_prompt_with_appended_silence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reference = Path(temp_dir) / "reference.wav"
            stereo = np.column_stack(
                [
                    np.full(100, 0.2, dtype=np.float32),
                    np.full(100, 0.4, dtype=np.float32),
                ]
            )
            sf.write(reference, stereo, 1000, subtype="FLOAT")
            config = {
                **MODULE.DEFAULT_CONFIG,
                "mode": "semantic_icl",
                "ref_audio": reference,
                "ref_text": "Exakter Text.",
                "icl_append_silence_ms": 50,
            }
            model = mock.Mock()
            model.model.create_voice_clone_prompt.return_value = {"prompt": "icl"}

            result = MODULE.load_voice_prompt(model, config)

            self.assertEqual(result, {"prompt": "icl"})
            kwargs = model.model.create_voice_clone_prompt.call_args.kwargs
            audio, sample_rate = kwargs["ref_audio"]
            self.assertEqual(sample_rate, 1000)
            self.assertEqual(audio.shape, (150,))
            np.testing.assert_allclose(audio[:100], 0.3, atol=1e-6)
            np.testing.assert_array_equal(audio[100:], np.zeros(50, dtype=np.float32))
            self.assertEqual(kwargs["ref_text"], "Exakter Text.")
            self.assertFalse(kwargs["x_vector_only_mode"])

    def test_write_mp3_wraps_encoder_errors(self):
        with mock.patch("soundfile.write", side_effect=RuntimeError("encoder failed")):
            with self.assertRaisesRegex(RuntimeError, "MP3 encoding failed"):
                MODULE.write_mp3(Path("out.mp3"), np.zeros(10, dtype=np.float32), 24000)

    def test_delete_parts_directory_rejects_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            output = directory / "result.mp3"
            expected_parts = directory / "result_parts"

            with self.assertRaisesRegex(RuntimeError, "completed MP3"):
                MODULE.delete_parts_directory(expected_parts, output)

            output.write_bytes(b"mp3")
            unexpected = directory / "other_parts"
            unexpected.mkdir()
            with self.assertRaisesRegex(RuntimeError, "unexpected parts directory"):
                MODULE.delete_parts_directory(unexpected, output)

            MODULE.delete_parts_directory(expected_parts, output)

            expected_parts.write_text("not a directory", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not a directory"):
                MODULE.delete_parts_directory(expected_parts, output)

    def test_audio_conversion_silence_trimming_loudness_and_fades(self):
        import torch

        tensor = torch.tensor([[0.1, -0.2]], dtype=torch.float64)
        converted = MODULE._to_float32_audio(tensor)
        self.assertEqual(converted.dtype, np.float32)
        np.testing.assert_allclose(converted, [0.1, -0.2])

        self.assertEqual(MODULE.edge_silence_samples(np.zeros(0), -50.0), (0, 0))
        self.assertEqual(MODULE.edge_silence_samples(np.zeros(10), -50.0), (10, 10))

        audio = np.concatenate(
            [
                np.zeros(100, dtype=np.float32),
                np.full(100, 0.2, dtype=np.float32),
                np.zeros(80, dtype=np.float32),
            ]
        )
        trimmed = MODULE.trim_excess_edge_silence(
            audio,
            1000,
            threshold_db=-50.0,
            max_leading_ms=10,
            max_trailing_ms=20,
        )
        self.assertEqual(len(trimmed), 130)
        self.assertEqual(MODULE.edge_silence_samples(trimmed, -50.0), (10, 20))

        adjusted, adjusted_rms = MODULE.match_chunk_loudness(
            np.full(20, 0.1, dtype=np.float32),
            0.2,
            threshold_db=-50.0,
            max_adjustment_db=3.0,
        )
        expected_level = 0.1 * 10 ** (3.0 / 20)
        np.testing.assert_allclose(adjusted, expected_level, rtol=1e-6)
        self.assertAlmostEqual(adjusted_rms, expected_level, places=6)

        faded = MODULE._apply_edge_fades(np.ones(10, dtype=np.float32), 1000, 3)
        self.assertEqual(float(faded[0]), 0.0)
        self.assertEqual(float(faded[-1]), 0.0)
        self.assertEqual(float(faded[4]), 1.0)

    def test_join_audio_chunks_handles_all_boundaries_empty_input_and_crossfade(self):
        empty = MODULE.join_audio_chunks([], [], 1000, MODULE.DEFAULT_CONFIG)
        self.assertEqual(empty.dtype, np.float32)
        self.assertEqual(len(empty), 0)

        first = np.full(10, 0.2, dtype=np.float32)
        second = np.full(10, 0.3, dtype=np.float32)
        pause_by_boundary = {
            "sentence": MODULE.DEFAULT_CONFIG["sentence_pause_ms"],
            "paragraph": MODULE.DEFAULT_CONFIG["paragraph_pause_ms"],
            "topic": MODULE.DEFAULT_CONFIG["topic_pause_ms"],
        }
        for boundary, pause_ms in pause_by_boundary.items():
            with self.subTest(boundary=boundary):
                config = {**MODULE.DEFAULT_CONFIG, "edge_fade_ms": 0}
                joined = MODULE.join_audio_chunks(
                    [first, second],
                    [MODULE.TextChunk("eins", boundary), MODULE.TextChunk("zwei", "end")],
                    1000,
                    config,
                )
                self.assertEqual(len(joined), 20 + pause_ms)

        crossfade_config = {
            **MODULE.DEFAULT_CONFIG,
            "edge_fade_ms": 0,
            "crossfade_ms": 5,
        }
        crossfaded = MODULE.join_audio_chunks(
            [first, second],
            [MODULE.TextChunk("eins", "end"), MODULE.TextChunk("zwei", "end")],
            1000,
            crossfade_config,
        )
        self.assertEqual(len(crossfaded), 15)
        self.assertEqual(crossfaded.dtype, np.float32)

    def test_streaming_generation_collects_audio_and_timings(self):
        model = mock.Mock()
        model.generate_voice_clone_streaming.return_value = iter(
            [
                (
                    np.array([0.1, 0.2], dtype=np.float32),
                    24000,
                    {"steps": 1, "prefill_ms": 2.0, "decode_s": 0.01, "ms_per_step": 10.0},
                ),
                (
                    np.array([0.3], dtype=np.float32),
                    24000,
                    {"steps": 2, "prefill_ms": 3.0, "decode_s": 0.02, "ms_per_step": 11.0},
                ),
            ]
        )
        config = {**MODULE.DEFAULT_CONFIG, "instruct": "Ruhig"}

        audio, sample_rate, timing = MODULE._generate_audio_chunk_streaming(
            model,
            {"voice": "prompt"},
            "Hallo.",
            config,
            run_warmup=True,
            seed=123,
        )

        np.testing.assert_allclose(audio, [0.1, 0.2, 0.3])
        self.assertEqual(sample_rate, 24000)
        self.assertEqual(timing["steps"], 2)
        self.assertEqual(timing["prefill_ms"], 3.0)
        self.assertEqual(timing["decode_s"], 0.02)
        self.assertGreaterEqual(timing["warmup_s"], 0.0)
        model.warmup.assert_called_once_with(prefill_len=100)
        kwargs = model.generate_voice_clone_streaming.call_args.kwargs
        self.assertEqual(kwargs["text"], "Hallo.")
        self.assertEqual(kwargs["instruct"], "Ruhig")
        self.assertEqual(kwargs["max_new_tokens"], config["max_new_tokens"])

    def test_streaming_generation_rejects_empty_audio_and_sample_rate_changes(self):
        config = {**MODULE.DEFAULT_CONFIG}
        model = mock.Mock()
        model.generate_voice_clone_streaming.return_value = iter([])
        with self.assertRaisesRegex(RuntimeError, "returned no audio"):
            MODULE._generate_audio_chunk_streaming(
                model, {}, "Hallo.", config, run_warmup=False, seed=1
            )

        model.generate_voice_clone_streaming.return_value = iter(
            [
                (np.ones(2, dtype=np.float32), 24000, {}),
                (np.ones(2, dtype=np.float32), 16000, {}),
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "sample rate changed"):
            MODULE._generate_audio_chunk_streaming(
                model, {}, "Hallo.", config, run_warmup=False, seed=1
            )

    def test_find_pause_cut_selects_nearest_pause_and_handles_no_candidate(self):
        active = np.full(100, 0.2, dtype=np.float32)
        silence = np.zeros(200, dtype=np.float32)
        audio = np.concatenate([active, silence, active, silence, active])

        cut = MODULE.find_pause_cut(
            audio,
            1000,
            expected_pause_start_sample=410,
            search_window_ms=50,
            min_pause_ms=100,
            lead_in_ms=10,
            threshold_db=-50.0,
        )

        self.assertIsNotNone(cut)
        self.assertGreaterEqual(cut.pause_start_sample, 390)
        self.assertLessEqual(cut.pause_start_sample, 410)
        self.assertEqual(cut.cut_sample, cut.pause_end_sample - 10)
        self.assertIsNone(
            MODULE.find_pause_cut(
                np.ones(100, dtype=np.float32),
                1000,
                expected_pause_start_sample=50,
                search_window_ms=50,
                min_pause_ms=20,
                lead_in_ms=0,
                threshold_db=-50.0,
            )
        )
        self.assertIsNone(
            MODULE.find_pause_cut(
                np.zeros(0, dtype=np.float32),
                0,
                expected_pause_start_sample=0,
                search_window_ms=0,
                min_pause_ms=20,
                lead_in_ms=0,
                threshold_db=-50.0,
            )
        )

    def test_text_preroll_calibrates_once_and_reuses_cut_for_later_chunks(self):
        calls = []
        base_timing = {
            "steps": 1,
            "prefill_ms": 1.0,
            "decode_s": 0.01,
            "ms_per_step": 10.0,
            "codec_decode_s": 0.01,
            "warmup_s": 0.2,
            "tts_wall_s": 0.5,
        }

        def original_generator(*args, **kwargs):
            calls.append((args[2], kwargs["run_warmup"]))
            if len(calls) == 1:
                audio = np.concatenate(
                    [np.full(100, 0.2, dtype=np.float32), np.zeros(200, dtype=np.float32)]
                )
            else:
                audio = np.concatenate(
                    [
                        np.full(100, 0.2, dtype=np.float32),
                        np.zeros(200, dtype=np.float32),
                        np.full(100, 0.1, dtype=np.float32),
                    ]
                )
            return audio, 1000, dict(base_timing)

        with tempfile.TemporaryDirectory() as temp_dir:
            generator = MODULE.TextPrerollChunkGenerator(
                original_generator,
                sentence="Kalibrierung",
                search_window_ms=50,
                min_pause_ms=100,
                lead_in_ms=0,
                debug_directory=Path(temp_dir),
                save_debug_wav=False,
            )
            config = {**MODULE.DEFAULT_CONFIG, "silence_threshold_db": -50.0}

            first_audio, _, first_timing = generator(
                mock.Mock(), {}, "Erstes Ziel.", config, mock.Mock(), run_warmup=True, seed=1
            )
            second_audio, _, second_timing = generator(
                mock.Mock(), {}, "Zweites Ziel.", config, mock.Mock(), run_warmup=False, seed=2
            )

        self.assertEqual(generator.sentence, "Kalibrierung.")
        self.assertEqual(
            [text for text, _ in calls],
            [
                "Kalibrierung.",
                "Kalibrierung.\n\nErstes Ziel.",
                "Kalibrierung.\n\nZweites Ziel.",
            ],
        )
        self.assertEqual([warmup for _, warmup in calls], [True, False, False])
        self.assertEqual(len(first_audio), 100)
        self.assertEqual(len(second_audio), 100)
        self.assertEqual(first_timing["tts_wall_s"], 1.0)
        self.assertEqual(second_timing["tts_wall_s"], 0.5)

    def test_text_preroll_accepts_low_level_audio_inside_boundary_pause(self):
        timing = {
            "steps": 1,
            "prefill_ms": 1.0,
            "decode_s": 0.01,
            "ms_per_step": 10.0,
            "codec_decode_s": 0.01,
            "warmup_s": 0.0,
            "tts_wall_s": 0.1,
        }
        calibration = np.concatenate(
            [np.full(100, 0.2, dtype=np.float32), np.zeros(100, dtype=np.float32)]
        )
        noisy_boundary = np.concatenate(
            [
                np.full(100, 0.2, dtype=np.float32),
                np.full(160, 0.005, dtype=np.float32),
                np.full(100, 0.2, dtype=np.float32),
            ]
        )
        self.assertIsNone(
            MODULE.find_pause_cut(
                noisy_boundary,
                1000,
                expected_pause_start_sample=100,
                search_window_ms=50,
                min_pause_ms=100,
                lead_in_ms=0,
                threshold_db=-50.0,
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            original_generator = mock.Mock(
                side_effect=[
                    (calibration, 1000, dict(timing)),
                    (noisy_boundary, 1000, dict(timing)),
                ]
            )
            wrapper = MODULE.TextPrerollChunkGenerator(
                original_generator,
                sentence="Kalibrierung.",
                search_window_ms=50,
                min_pause_ms=100,
                lead_in_ms=0,
                debug_directory=Path(temp_dir),
                save_debug_wav=False,
            )
            audio, _, _ = wrapper(
                mock.Mock(),
                {},
                "Ziel.",
                {**MODULE.DEFAULT_CONFIG, "silence_threshold_db": -50.0},
                mock.Mock(),
                run_warmup=False,
                seed=1,
            )

        np.testing.assert_array_equal(audio, np.full(100, 0.2, dtype=np.float32))

    def test_text_preroll_reports_cut_failure_and_sample_rate_change(self):
        timing = {
            "steps": 1,
            "prefill_ms": 1.0,
            "decode_s": 0.01,
            "ms_per_step": 10.0,
            "codec_decode_s": 0.01,
            "warmup_s": 0.0,
            "tts_wall_s": 0.1,
        }
        calibration = np.concatenate(
            [np.full(100, 0.2, dtype=np.float32), np.zeros(100, dtype=np.float32)]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            failing_generator = mock.Mock(
                side_effect=[
                    (calibration, 1000, dict(timing)),
                    (np.ones(200, dtype=np.float32), 1000, dict(timing)),
                ]
            )
            wrapper = MODULE.TextPrerollChunkGenerator(
                failing_generator,
                sentence="Kalibrierung.",
                search_window_ms=50,
                min_pause_ms=50,
                lead_in_ms=0,
                debug_directory=directory,
                save_debug_wav=False,
            )
            with mock.patch.object(MODULE, "find_pause_cut", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "no suitable text-preroll pause"):
                    wrapper(
                        mock.Mock(),
                        {},
                        "Ziel.",
                        MODULE.DEFAULT_CONFIG,
                        mock.Mock(),
                        run_warmup=False,
                        seed=1,
                    )
            self.assertTrue(
                (directory / "text-preroll-cut-failed-chunk-001.wav").is_file()
            )

            rate_changing_generator = mock.Mock(
                side_effect=[
                    (calibration, 1000, dict(timing)),
                    (np.ones(200, dtype=np.float32), 2000, dict(timing)),
                ]
            )
            wrapper = MODULE.TextPrerollChunkGenerator(
                rate_changing_generator,
                sentence="Kalibrierung.",
                search_window_ms=50,
                min_pause_ms=50,
                lead_in_ms=0,
                debug_directory=directory,
                save_debug_wav=False,
            )
            with self.assertRaisesRegex(RuntimeError, "sample rate changed"):
                wrapper(
                    mock.Mock(),
                    {},
                    "Ziel.",
                    MODULE.DEFAULT_CONFIG,
                    mock.Mock(),
                    run_warmup=False,
                    seed=1,
                )

    def test_generate_audio_chunk_warmup_sampling_ref_codes_and_empty_tokens(self):
        import torch

        class FakeTokenizer:
            def __init__(self):
                self.inputs = []

            def decode(self, payload):
                codes = payload["audio_codes"].squeeze(0).clone()
                self.inputs.append(codes)
                return [np.arange(codes.shape[0] * 10, dtype=np.float32)], 24000

        tokenizer = FakeTokenizer()
        prepared_model = mock.Mock()
        prepared_model.speech_tokenizer = tokenizer
        model = mock.Mock()
        ref_codes = torch.tensor([[9], [8]])
        model._prepare_generation.return_value = (
            prepared_model,
            mock.Mock(),
            mock.Mock(),
            torch.zeros((1, 3, 2)),
            mock.Mock(),
            mock.Mock(),
            mock.Mock(),
            ref_codes,
        )
        timing = {
            "steps": 3,
            "prefill_ms": 1.0,
            "decode_s": 0.03,
            "ms_per_step": 10.0,
        }
        fast_generate = mock.Mock(
            side_effect=[
                (torch.tensor([[7]]), dict(timing)),
                (torch.tensor([[1], [2], [3]]), dict(timing)),
            ]
        )
        config = {**MODULE.DEFAULT_CONFIG, "codec_context_frames": 0}
        context_state = {"codes": torch.tensor([[99]])}

        audio, sample_rate, result_timing = MODULE._generate_audio_chunk(
            model,
            {},
            "Hallo.",
            config,
            fast_generate,
            run_warmup=True,
            seed=123,
            codec_context_state=context_state,
        )

        model.warmup.assert_called_once_with(prefill_len=100)
        self.assertEqual(fast_generate.call_count, 2)
        self.assertEqual(
            fast_generate.call_args_list[0].kwargs["max_new_tokens"],
            config["warmup_max_new_tokens"],
        )
        self.assertEqual(
            fast_generate.call_args_list[1].kwargs["max_new_tokens"],
            config["max_new_tokens"],
        )
        self.assertEqual(tokenizer.inputs[0].flatten().tolist(), [9, 8, 1, 2, 3])
        self.assertEqual(len(audio), 30)
        self.assertEqual(sample_rate, 24000)
        self.assertIsNone(context_state["codes"])
        self.assertEqual(result_timing["codec_context_frames"], 0)
        self.assertGreaterEqual(result_timing["warmup_s"], 0.0)

        empty_generator = mock.Mock(
            return_value=(torch.empty((0, 1), dtype=torch.long), dict(timing))
        )
        with self.assertRaisesRegex(RuntimeError, "returned no tokens"):
            MODULE._generate_audio_chunk(
                model,
                {},
                "Hallo.",
                config,
                empty_generator,
                run_warmup=False,
                seed=123,
            )

    def test_text_helpers_reject_invalid_inputs_and_preserve_boundaries(self):
        with self.assertRaisesRegex(ValueError, "max_chars"):
            MODULE.chunk_text("Hallo.", max_chars=0)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            MODULE.chunk_text("   ")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            MODULE.chunk_text("Hallo.", sentences_per_chunk=0)

        self.assertTrue(MODULE._looks_like_heading("# Überschrift"))
        self.assertTrue(MODULE._looks_like_heading("2. Neues Thema"))
        self.assertFalse(MODULE._looks_like_heading("Das ist ein normaler Satz."))
        with self.assertRaisesRegex(ValueError, "without breaking a word"):
            MODULE._split_long_sentence("A" * 101, 100)

        with self.assertRaisesRegex(ValueError, "semantic sizes"):
            MODULE.semantic_chunk_text("Hallo.", min_chars=20, target_chars=10, max_chars=30)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            MODULE.semantic_chunk_text("   ")

        semantic_config = {**MODULE.DEFAULT_CONFIG, "mode": "semantic"}
        with self.assertRaisesRegex(ValueError, "--sentences"):
            MODULE.build_text_chunks("Hallo.", semantic_config, sentences_per_chunk=1)
        with self.assertRaisesRegex(ValueError, "--padding"):
            MODULE.build_text_chunks("Hallo.", semantic_config, padding=True)

        legacy_config = {
            **MODULE.DEFAULT_CONFIG,
            "mode": "legacy",
            "legacy_chunk_chars": 10,
        }
        legacy = MODULE.build_text_chunks("Eins.\n\nZwei.", legacy_config)
        self.assertEqual([chunk.boundary_after for chunk in legacy], ["paragraph", "end"])

    def test_prepare_generation_text_disabled_and_idempotent(self):
        disabled = {**MODULE.DEFAULT_CONFIG, "append_chunk_end_padding": False}
        self.assertEqual(MODULE.prepare_generation_text("Hallo.  ", disabled), "Hallo.")

        enabled = {**MODULE.DEFAULT_CONFIG, "append_chunk_end_padding": True}
        prepared = MODULE.prepare_generation_text("Hallo.", enabled)
        self.assertEqual(prepared, "Hallo.\n\n.")
        self.assertEqual(MODULE.prepare_generation_text(prepared, enabled), prepared)

    def test_generate_mp3_uses_streaming_backend_and_incrementing_seeds(self):
        import faster_qwen3_tts

        timing = {
            "steps": 1,
            "prefill_ms": 1.0,
            "decode_s": 0.01,
            "ms_per_step": 10.0,
            "codec_decode_s": 0.0,
            "warmup_s": 0.0,
            "tts_wall_s": 0.01,
        }
        generated = [
            (np.full(800, 0.1, dtype=np.float32), 8000, dict(timing)),
            (np.full(800, 0.2, dtype=np.float32), 8000, dict(timing)),
        ]
        config = {
            **MODULE.DEFAULT_CONFIG,
            "speaker": Path("voice.pt"),
            "generation_api": "streaming",
            "text_preroll_enabled": False,
            "append_chunk_end_padding": False,
            "save_wav_parts": False,
            "seed": 40,
            "seed_strategy": "increment",
            "edge_fade_ms": 0,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "streaming.mp3"
            fake_model = mock.Mock()
            with (
                mock.patch.object(
                    faster_qwen3_tts.FasterQwen3TTS,
                    "from_pretrained",
                    return_value=fake_model,
                ),
                mock.patch.object(MODULE, "load_voice_prompt", return_value={}),
                mock.patch.object(
                    MODULE,
                    "_generate_audio_chunk_streaming",
                    side_effect=generated,
                ) as streaming_generator,
                mock.patch.object(MODULE, "_generate_audio_chunk") as non_streaming_generator,
                mock.patch("builtins.print"),
            ):
                report = MODULE.generate_mp3(
                    [
                        MODULE.TextChunk("Erster.", "paragraph"),
                        MODULE.TextChunk("Zweiter.", "end"),
                    ],
                    output,
                    config,
                )

            self.assertTrue(output.is_file())
            self.assertEqual(streaming_generator.call_count, 2)
            non_streaming_generator.assert_not_called()
            self.assertEqual(
                [call.kwargs["seed"] for call in streaming_generator.call_args_list],
                [40, 41],
            )
            self.assertEqual(
                [call.args[2] for call in streaming_generator.call_args_list],
                ["Erster.", "Zweiter."],
            )
            self.assertEqual([chunk["seed"] for chunk in report["chunks"]], [40, 41])
            self.assertEqual(report["generation_api"], "streaming")
            self.assertEqual(report["chunk_count"], 2)

    def test_generate_mp3_rejects_sample_rate_changes_and_keeps_parts_on_failure(self):
        import faster_qwen3_tts

        timing = {
            "steps": 1,
            "prefill_ms": 1.0,
            "decode_s": 0.01,
            "ms_per_step": 10.0,
            "codec_decode_s": 0.01,
            "warmup_s": 0.0,
            "tts_wall_s": 0.01,
        }
        base_config = {
            **MODULE.DEFAULT_CONFIG,
            "speaker": Path("voice.pt"),
            "text_preroll_enabled": False,
            "append_chunk_end_padding": False,
            "edge_fade_ms": 0,
        }
        fake_model = mock.Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            output = directory / "rate-change.mp3"
            with (
                mock.patch.object(
                    faster_qwen3_tts.FasterQwen3TTS,
                    "from_pretrained",
                    return_value=fake_model,
                ),
                mock.patch.object(MODULE, "load_voice_prompt", return_value={}),
                mock.patch.object(
                    MODULE,
                    "_generate_audio_chunk",
                    side_effect=[
                        (np.ones(10, dtype=np.float32), 24000, dict(timing)),
                        (np.ones(10, dtype=np.float32), 16000, dict(timing)),
                    ],
                ),
                mock.patch("builtins.print"),
            ):
                with self.assertRaisesRegex(RuntimeError, "sample rate changed between chunks"):
                    MODULE.generate_mp3(["Eins.", "Zwei."], output, base_config)

            failed_output = directory / "encode-failure.mp3"
            parts_directory = directory / "encode-failure_parts"
            with (
                mock.patch.object(
                    faster_qwen3_tts.FasterQwen3TTS,
                    "from_pretrained",
                    return_value=fake_model,
                ),
                mock.patch.object(MODULE, "load_voice_prompt", return_value={}),
                mock.patch.object(
                    MODULE,
                    "_generate_audio_chunk",
                    return_value=(np.ones(100, dtype=np.float32), 1000, dict(timing)),
                ),
                mock.patch.object(MODULE, "write_mp3", side_effect=RuntimeError("encode failed")),
                mock.patch("builtins.print"),
            ):
                with self.assertRaisesRegex(RuntimeError, "encode failed"):
                    MODULE.generate_mp3(["Eins."], failed_output, base_config)

            self.assertTrue((parts_directory / "teil-001.wav").is_file())

    def test_generate_mp3_rejects_empty_chunks(self):
        import faster_qwen3_tts

        config = {
            **MODULE.DEFAULT_CONFIG,
            "speaker": Path("voice.pt"),
            "text_preroll_enabled": False,
            "save_wav_parts": False,
        }
        with (
            mock.patch.object(
                faster_qwen3_tts.FasterQwen3TTS,
                "from_pretrained",
                return_value=mock.Mock(),
            ),
            mock.patch.object(MODULE, "load_voice_prompt", return_value={}),
            mock.patch("builtins.print"),
        ):
            with self.assertRaisesRegex(RuntimeError, "no audio chunks"):
                MODULE.generate_mp3([], Path("unused.mp3"), config)

    def test_main_resolves_overrides_builds_chunks_and_writes_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            speaker = directory / "voice.pt"
            speaker.touch()
            config_path = directory / "config.json"
            _write_config(config_path, speaker, mode="legacy", save_wav_parts=True)
            output = directory / "result.mp3"
            metrics = directory / "reports" / "metrics.json"
            report = {"status": "ok", "chunk_count": 2}

            with mock.patch.object(MODULE, "generate_mp3", return_value=report) as generate:
                result = MODULE.main(
                    [
                        "--text",
                        "Eins. Zwei.",
                        "--output",
                        str(output),
                        "--config",
                        str(config_path),
                        "--mode",
                        "legacy",
                        "--characters",
                        "500",
                        "--sentences",
                        "1",
                        "--no-wav-parts",
                        "--metrics",
                        str(metrics),
                    ]
                )

            self.assertEqual(result, 0)
            chunks, passed_output, passed_config = generate.call_args.args
            self.assertEqual([chunk.text for chunk in chunks], ["Eins.", "Zwei."])
            self.assertEqual(passed_output, output)
            self.assertEqual(passed_config["legacy_chunk_chars"], 500)
            self.assertFalse(passed_config["save_wav_parts"])
            self.assertEqual(json.loads(metrics.read_text(encoding="utf-8")), report)

    def test_main_applies_clear_markdown_before_chunking(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            speaker = directory / "voice.pt"
            speaker.touch()
            config_path = directory / "config.json"
            _write_config(
                config_path,
                speaker,
                mode="legacy",
                speak_numbered_lists=False,
                clear_markdown=False,
            )
            output = directory / "result.mp3"

            with mock.patch.object(
                MODULE,
                "generate_mp3",
                return_value={"status": "ok"},
            ) as generate:
                result = MODULE.main(
                    [
                        "--text",
                        "Termin: 01.08.2026.\n"
                        "Quelle: https://example.org/quelle\n"
                        "Ablauf: 1. starten.",
                        "--output",
                        str(output),
                        "--config",
                        str(config_path),
                        "--clear_markdown",
                    ]
                )

            self.assertEqual(result, 0)
            chunks, passed_output, _ = generate.call_args.args
            self.assertEqual(
                [chunk.text for chunk in chunks],
                ["Termin: Erster August 2026.\nAblauf: erstens starten."],
            )
            self.assertEqual(passed_output, output)

    def test_main_uses_configured_clear_markdown_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            speaker = directory / "voice.pt"
            speaker.touch()
            config_path = directory / "config.json"
            _write_config(
                config_path,
                speaker,
                mode="legacy",
                speak_numbered_lists=False,
                clear_markdown=True,
            )

            with mock.patch.object(
                MODULE,
                "generate_mp3",
                return_value={"status": "ok"},
            ) as generate:
                result = MODULE.main(
                    [
                        "--text",
                        "Datum: 31.02.2026\nText bleibt.",
                        "--output",
                        str(directory / "result.mp3"),
                        "--config",
                        str(config_path),
                    ]
                )

            self.assertEqual(result, 0)
            chunks, _, passed_config = generate.call_args.args
            self.assertEqual([chunk.text for chunk in chunks], ["Text bleibt."])
            self.assertTrue(passed_config["clear_markdown"])

    def test_main_no_clear_markdown_overrides_configured_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            speaker = directory / "voice.pt"
            speaker.touch()
            config_path = directory / "config.json"
            _write_config(
                config_path,
                speaker,
                mode="legacy",
                speak_numbered_lists=False,
                clear_markdown=True,
            )

            with mock.patch.object(
                MODULE,
                "generate_mp3",
                return_value={"status": "ok"},
            ) as generate:
                result = MODULE.main(
                    [
                        "--text",
                        "Datum: 31.02.2026\nText bleibt.",
                        "--output",
                        str(directory / "result.mp3"),
                        "--config",
                        str(config_path),
                        "--no-clear-markdown",
                    ]
                )

            self.assertEqual(result, 0)
            chunks, _, passed_config = generate.call_args.args
            self.assertEqual(
                [chunk.text for chunk in chunks],
                ["Datum: 31.02.2026\nText bleibt."],
            )
            self.assertFalse(passed_config["clear_markdown"])

    def test_main_prints_and_writes_cleaned_text_without_loading_tts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cleaned_path = Path(temp_dir) / "nested" / "cleaned.md"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch("sys.stdout", stdout),
                mock.patch("sys.stderr", stderr),
                mock.patch.object(MODULE, "load_config") as load_config,
                mock.patch.object(MODULE, "generate_mp3") as generate,
            ):
                result = MODULE.main(
                    [
                        "--text",
                        "Termin: 01.08.2026. Quelle: https://example.org",
                        "--clear_markdown",
                        "--print_cleaned_text",
                        "--write_cleaned_text",
                        str(cleaned_path),
                    ]
                )

            expected = "Termin: Erster August 2026. Quelle:"
            self.assertEqual(result, 0)
            self.assertEqual(stdout.getvalue(), f"{expected}\n")
            self.assertIn(str(cleaned_path), stderr.getvalue())
            self.assertEqual(cleaned_path.read_text(encoding="utf-8"), expected)
            load_config.assert_not_called()
            generate.assert_not_called()

    def test_main_cleaned_text_output_uses_configured_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            config_path = directory / "config.json"
            config_path.write_text(
                json.dumps({"clear_markdown": True}),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with (
                mock.patch("sys.stdout", stdout),
                mock.patch.object(MODULE, "load_config") as load_config,
                mock.patch.object(MODULE, "generate_mp3") as generate,
            ):
                result = MODULE.main(
                    [
                        "--text",
                        "Datum: 31.02.2026\nText bleibt.",
                        "--config",
                        str(config_path),
                        "--print_cleaned_text",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(stdout.getvalue(), "Text bleibt.\n")
            load_config.assert_not_called()
            generate.assert_not_called()

    def test_main_rejects_cleaned_text_output_when_effective_default_is_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps({"clear_markdown": False}),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit) as raised:
                MODULE.main(
                    [
                        "--text",
                        "Hallo.",
                        "--config",
                        str(config_path),
                        "--print_cleaned_text",
                    ]
                )

            self.assertEqual(raised.exception.code, 2)

    def test_main_rejects_input_removed_entirely_by_clear_markdown(self):
        with self.assertRaises(SystemExit) as raised:
            MODULE.main(
                [
                    "--text",
                    "```python\nprint('entfernen')\n```",
                    "--output",
                    "out.mp3",
                    "--clear_markdown",
                ]
            )

        self.assertEqual(raised.exception.code, 2)

    def test_main_converts_validation_errors_to_parser_exit(self):
        with self.assertRaises(SystemExit) as raised:
            MODULE.main(["--text", "   ", "--output", "out.mp3"])
        self.assertEqual(raised.exception.code, 2)

    def test_semantic_chunking_matches_representative_golden_result(self):
        text = (
            "# Auftakt\n\n"
            "Dr. Müller erklärt den ersten Gedanken ausführlich. "
            "Danach folgt ein zweiter vollständiger Satz.\n\n"
            "# Wechsel\n\n"
            "1. Der erste Punkt bleibt zusammen. "
            "2. Der zweite Punkt schließt das Thema ab."
        )
        spoken = MODULE.normalize_numbered_list_markers(text, "German")

        chunks = MODULE.semantic_chunk_text(
            spoken,
            target_chars=105,
            min_chars=60,
            max_chars=145,
        )

        self.assertEqual(
            chunks,
            [
                MODULE.TextChunk(
                    "Auftakt\n\nDr. Müller erklärt den ersten Gedanken ausführlich. "
                    "Danach folgt ein zweiter vollständiger Satz.",
                    "topic",
                ),
                MODULE.TextChunk(
                    "Wechsel\n\nErstens, Der erste Punkt bleibt zusammen. "
                    "2. Der zweite Punkt schließt das Thema ab.",
                    "end",
                ),
            ],
        )

    def test_join_audio_chunks_matches_exact_pcm_for_pause_and_crossfade(self):
        pause_config = {
            **MODULE.DEFAULT_CONFIG,
            "sentence_pause_ms": 4,
            "edge_fade_ms": 0,
        }
        paused = MODULE.join_audio_chunks(
            [
                np.array([1.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.5, 0.5], dtype=np.float32),
            ],
            [MODULE.TextChunk("eins", "sentence"), MODULE.TextChunk("zwei", "end")],
            1000,
            pause_config,
        )
        np.testing.assert_array_equal(
            paused,
            np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5], dtype=np.float32),
        )

        crossfade_config = {
            **MODULE.DEFAULT_CONFIG,
            "crossfade_ms": 2,
            "edge_fade_ms": 0,
        }
        crossfaded = MODULE.join_audio_chunks(
            [np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32)],
            [MODULE.TextChunk("eins", "end"), MODULE.TextChunk("zwei", "end")],
            1000,
            crossfade_config,
        )
        np.testing.assert_array_equal(
            crossfaded,
            np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

    def test_load_config_allows_disabled_preroll_and_requires_speaker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            speaker = directory / "voice.pt"
            speaker.touch()
            streaming_config = directory / "streaming.json"
            _write_config(
                streaming_config,
                speaker,
                generation_api="streaming",
                text_preroll_enabled=False,
            )

            config = MODULE.load_config(streaming_config)
            self.assertEqual(config["generation_api"], "streaming")
            self.assertFalse(config["text_preroll_enabled"])

            missing_speaker = directory / "empty-speaker.json"
            missing_speaker.write_text(json.dumps({"speaker": ""}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must point to a .pt"):
                MODULE.load_config(missing_speaker)

    def test_german_list_ordinals_cover_limits_and_unhandled_values(self):
        self.assertEqual(MODULE._german_list_ordinal(20), "zwanzigstens")
        self.assertEqual(MODULE._german_list_ordinal(42), "zweiundvierzigstens")
        self.assertEqual(MODULE._german_list_ordinal(100), "hundertstens")
        self.assertIsNone(MODULE._german_list_ordinal(101))
        self.assertEqual(
            MODULE.normalize_numbered_list_markers("101. Unverändert", "German"),
            "101. Unverändert",
        )

    def test_active_rms_and_preroll_constructor_handle_silence_and_empty_sentence(self):
        self.assertEqual(MODULE._active_rms(np.zeros(0, dtype=np.float32), -50.0), 0.0)
        self.assertEqual(MODULE._active_rms(np.zeros(10, dtype=np.float32), -50.0), 0.0)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            MODULE.TextPrerollChunkGenerator(
                mock.Mock(),
                sentence="   ",
                search_window_ms=50,
                min_pause_ms=20,
                lead_in_ms=0,
                debug_directory=Path("debug"),
                save_debug_wav=False,
            )

    def test_delete_parts_directory_rejects_symlink_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            output = directory / "result.mp3"
            output.write_bytes(b"mp3")
            parts = directory / "result_parts"
            with mock.patch.object(Path, "is_symlink", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "symlinked parts directory"):
                    MODULE.delete_parts_directory(parts, output)

@pytest.mark.xfail(
    strict=True,
    reason="load_xvector_prompt calls .to() before validating the loaded object type",
)
def test_load_xvector_prompt_reports_non_tensor_as_value_error(tmp_path):
    import torch

    embedding_path = tmp_path / "not-a-tensor.pt"
    torch.save({"embedding": [0.0] * 2048}, embedding_path)

    with pytest.raises(ValueError, match="2048-element speaker embedding"):
        MODULE.load_xvector_prompt(embedding_path, "cpu")


@pytest.mark.xfail(
    strict=True,
    reason="--characters is not applied when legacy mode comes only from config",
)
def test_main_applies_characters_to_legacy_mode_selected_by_config(tmp_path, monkeypatch):
    speaker = tmp_path / "voice.pt"
    speaker.touch()
    config_path = tmp_path / "config.json"
    _write_config(config_path, speaker, mode="legacy")
    captured = {}

    def fake_generate(chunks, output, config):
        captured["config"] = config
        return {"chunk_count": len(chunks)}

    monkeypatch.setattr(MODULE, "generate_mp3", fake_generate)
    result = MODULE.main(
        [
            "--text",
            "Eins. Zwei.",
            "--output",
            str(tmp_path / "result.mp3"),
            "--config",
            str(config_path),
            "--characters",
            "300",
        ]
    )

    assert result == 0
    assert captured["config"]["legacy_chunk_chars"] == 300


@pytest.mark.xfail(
    strict=True,
    reason="write_mp3 writes directly to the destination instead of replacing it atomically",
)
def test_write_mp3_preserves_existing_output_when_encoding_fails(tmp_path, monkeypatch):
    output = tmp_path / "existing.mp3"
    output.write_bytes(b"original output")

    def truncate_then_fail(path, *args, **kwargs):
        Path(path).write_bytes(b"")
        raise RuntimeError("encoder failed")

    monkeypatch.setattr("soundfile.write", truncate_then_fail)
    with pytest.raises(RuntimeError, match="MP3 encoding failed"):
        MODULE.write_mp3(output, np.zeros(10, dtype=np.float32), 24000)

    assert output.read_bytes() == b"original output"


if __name__ == "__main__":
    unittest.main()
