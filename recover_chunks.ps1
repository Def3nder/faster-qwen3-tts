@'
from pathlib import Path
import soundfile as sf

from generate import generate_mp3_with_embedding as g

parts = next(
    p for p in Path("input").glob("*_parts")
    if (p / "text-preroll-cut-failed-chunk-046.wav").exists()
)
input_path = parts.with_name(parts.name.removesuffix("_parts") + ".md")
output_path = input_path.with_name(input_path.stem + "_recovered.mp3")

config = g.load_config(Path("generate/config.json"))

text = g.resolve_text(None, input_path)
if config["clear_markdown"]:
    text = g.clear_markdown_text(text)
chunks = g.build_text_chunks(text, config)
assert len(chunks) == 46

calibration, sr = sf.read(
    parts / "text-preroll-calibration.wav",
    dtype="float32",
    always_2d=False,
)
failed, failed_sr = sf.read(
    parts / "text-preroll-cut-failed-chunk-046.wav",
    dtype="float32",
    always_2d=False,
)
assert sr == failed_sr

_, trailing = g.edge_silence_samples(
    calibration,
    config["silence_threshold_db"],
)
expected = max(1, len(calibration) - trailing)

relaxed_db = min(
    -40.0,
    config["silence_threshold_db"] + 10.0,
)
cut = g.find_pause_cut(
    failed,
    sr,
    expected_pause_start_sample=expected,
    search_window_ms=config["text_preroll_search_window_ms"],
    min_pause_ms=config["text_preroll_min_pause_ms"],
    lead_in_ms=config["text_preroll_lead_in_ms"],
    threshold_db=relaxed_db,
)
if cut is None:
    raise RuntimeError("Preroll-Pause weiterhin nicht erkannt")

chunk46 = failed[cut.cut_sample:]

first, first_sr = sf.read(
    parts / "teil-001.wav",
    dtype="float32",
    always_2d=False,
)
assert first_sr == sr

reference_rms = g._active_rms(
    first,
    config["silence_threshold_db"],
)
chunk46 = g.trim_excess_edge_silence(
    chunk46,
    sr,
    threshold_db=config["silence_threshold_db"],
    max_leading_ms=config["max_leading_silence_ms"],
    max_trailing_ms=config["max_trailing_silence_ms"],
)
chunk46, _ = g.match_chunk_loudness(
    chunk46,
    reference_rms,
    threshold_db=config["silence_threshold_db"],
    max_adjustment_db=config["loudness_match_max_db"],
)
g.write_wav(parts / "teil-046.wav", chunk46, sr)

audio_chunks = []
for index in range(1, 47):
    audio, part_sr = sf.read(
        parts / f"teil-{index:03d}.wav",
        dtype="float32",
        always_2d=False,
    )
    assert part_sr == sr
    audio_chunks.append(g._to_float32_audio(audio))

combined = g.join_audio_chunks(audio_chunks, chunks, sr, config)
g.write_mp3(output_path, combined, sr)

print(f"Schnitt bei {cut.cut_sample / sr:.3f}s")
print(f"Erzeugt: {output_path.resolve()}")
'@ | .\.venv\Scripts\python.exe -