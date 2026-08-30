import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
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


if __name__ == "__main__":
    unittest.main()
