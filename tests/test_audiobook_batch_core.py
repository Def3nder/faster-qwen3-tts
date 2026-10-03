from __future__ import annotations

import json
from pathlib import Path

from generate.audiobook_batch.core import (
    SpeechOptions,
    assign_output_names,
    build_segments,
    build_toc_anchor_segments,
    create_batch_config,
    extract_footnotes,
    load_pronunciation_csv,
    load_project_replacements,
    numbers_to_words,
    normalize_output_title,
    paragraph_start,
    parse_toc_anchors,
    prepare_segments_for_speech,
    set_segment_range,
)


def test_h3_segments_include_title_and_offer_intro_separately():
    text = "# Buch\n\nVorspann.\n\n## A\n\n### Eins\n\nText eins.\n\n### Zwei\n\nText zwei.\n"
    segments = build_segments(text, 3)

    assert [(item.kind, item.title) for item in segments] == [
        ("intro", "Einleitung"),
        ("chapter", "Eins"),
        ("chapter", "Zwei"),
    ]
    assert not segments[0].included_by_default
    assert segments[1].source_text.startswith("### Eins")


def test_manual_split_snaps_to_paragraph_and_creates_parts():
    text = "### Kapitel\n\nErster Absatz.\n\nZweiter Absatz.\n"
    cursor = text.index("Zweiter") + 3
    cut = paragraph_start(text, cursor)
    segments = build_segments(text, 3, [cut])

    assert [segment.title for segment in segments] == ["Kapitel", "Kapitel – Teil 2"]
    assert segments[1].source_text.startswith("Zweiter")


def test_toc_anchor_links_create_chapters_without_markdown_headings():
    text = (
        "# Inhalt\n\n"
        "- [Erstes Kapitel](#a-1)\n"
        "- [Zweites Kapitel](#a-2)\n\n"
        "Vorspann.\n\n"
        '<a id="a-1"></a>\n\nErster Text.\n\n'
        '<a id="a-2"></a>\n\nZweiter Text.\n'
    )

    anchors = parse_toc_anchors(text)
    segments = build_toc_anchor_segments(text)

    assert [(anchor.identifier, anchor.title) for anchor in anchors] == [
        ("a-1", "Erstes Kapitel"),
        ("a-2", "Zweites Kapitel"),
    ]
    assert [(segment.kind, segment.title) for segment in segments] == [
        ("intro", "Einleitung"),
        ("chapter", "Erstes Kapitel"),
        ("chapter", "Zweites Kapitel"),
    ]
    assert not segments[0].included_by_default
    assert "Erster Text" in segments[1].source_text
    assert "Zweiter Text" not in segments[1].source_text
    assert "Zweiter Text" in segments[2].source_text


def test_user_selected_range_replaces_suggested_segment_text():
    text = "### Kapitel\n\nErster Absatz.\n\nZweiter Absatz.\n"
    segment = build_segments(text, 3)[0]
    start = text.index("Erster")
    end = text.index("Zweiter") - 2

    set_segment_range(segment, text, start, end)

    assert segment.start == start
    assert segment.end == end
    assert segment.source_text == "Erster Absatz."


def test_user_selected_range_rejects_empty_text():
    text = "### Kapitel\n\nText"
    segment = build_segments(text, 3)[0]
    start = text.index("\n")
    end = text.index("Text")

    try:
        set_segment_range(segment, text, start, end)
    except ValueError as error:
        assert "keinen vorlesbaren Text" in str(error)
    else:
        raise AssertionError("Leere Auswahl wurde unerwartet akzeptiert")


def test_footnote_is_spoken_once_directly_after_first_paragraph():
    text = "### Kapitel\n\nEin Satz.[^1]\n\nNoch einmal.[^1]\n\n[^1]: Eine wichtige Erklärung."
    segments = build_segments(text, 3)
    prepared = prepare_segments_for_speech(text, segments, SpeechOptions(speak_footnotes=True))[0]

    assert "Ein Satz.\n\nAnmerkung. Eine wichtige Erklärung." in prepared
    assert prepared.count("Eine wichtige Erklärung") == 1
    assert "[^1]" not in prepared


def test_footnotes_can_be_disabled():
    text = "### Kapitel\n\nEin Satz.[^x]\n\n[^x]: Nicht vorlesen."
    segments = build_segments(text, 3)
    prepared = prepare_segments_for_speech(text, segments, SpeechOptions(speak_footnotes=False))[0]

    assert "Nicht vorlesen" not in prepared
    assert "Ein Satz." in prepared


def test_html_anchors_and_images_are_removed_from_spoken_text():
    text = (
        '### Kapitel\n\n<a id="a-vier"></a>Vorher '
        '![image](images/arrow.jpg) Nachher <em>wichtig</em>. '
        '<!-- unsichtbarer Hinweis -->\n\n**Mehrzeilig\nformatiert**'
    )
    segments = build_segments(text, 3)
    prepared = prepare_segments_for_speech(text, segments, SpeechOptions())[0]

    assert '<a id="a-vier">' not in prepared
    assert "image" not in prepared
    assert "arrow.jpg" not in prepared
    assert "<em>" not in prepared
    assert "unsichtbarer Hinweis" not in prepared
    assert "Vorher Nachher wichtig." in prepared
    assert "**" not in prepared
    assert "Mehrzeilig\nformatiert" in prepared


def test_footnote_is_self_contained_in_each_output_file():
    text = "### Eins\n\nErster Verweis.[^x]\n\n### Zwei\n\nZweiter Verweis.[^x]\n\n[^x]: Gemeinsame Fußnote."
    segments = build_segments(text, 3)
    prepared = prepare_segments_for_speech(text, segments, SpeechOptions(speak_footnotes=True))

    assert "Gemeinsame Fußnote" in prepared[0]
    assert "Gemeinsame Fußnote" in prepared[1]


def test_lists_numbers_project_rules_and_phonetic_dictionary():
    directory = Path("generate/audiobook_batch")
    dictionary = directory / ".test-aussprache.csv"
    rules = directory / ".test-regeln.csv"
    try:
        dictionary.write_text(
            "Schreibweise;Aussprache;Phonetisch;Hinweis\nThiaoouba;Tiauba;Ti-a-u-ba;Test\n",
            encoding="utf-8",
        )
        rules.write_text("Suche;Ersatz;Hinweis\nX3;X drei;Test\n", encoding="utf-8")
        text = "### Test\n\n1. Thiaoouba hat X3 im Jahr 1987."
        segments = build_segments(text, 3)
        options = SpeechOptions(
            pronunciation_entries=load_pronunciation_csv(dictionary),
            project_replacements=load_project_replacements(rules),
        )
        prepared = prepare_segments_for_speech(text, segments, options)[0]
    finally:
        dictionary.unlink(missing_ok=True)
        rules.unlink(missing_ok=True)

    assert "Erstens," in prepared
    assert "Ti-a-u-ba" in prepared
    assert "X drei" in prepared
    assert "neunzehn-hundert-sieben-und-achtzig" in prepared


def test_output_names_only_number_selected_segments():
    text = "Vorspann\n\n### Äpfel & Öl\n\nText\n\n### Ende\n\nText"
    segments = build_segments(text, 3)
    assign_output_names(segments, [False, True, True])

    assert [segment.output_name for segment in segments] == [
        "—",
        "001_Äpfel_&_Öl.mp3",
        "002_Ende.mp3",
    ]


def test_editable_output_title_keeps_automatic_number_and_extension():
    text = "### Eins\n\nText\n\n### Zwei\n\nText"
    segments = build_segments(text, 3)
    custom_title = normalize_output_title("009_Mein neuer / Titel.mp3")

    assign_output_names(segments, [True, True], {segments[1].key: custom_title})

    assert custom_title == "Mein_neuer_Titel"
    assert normalize_output_title("1. Kapitel") == "1_Kapitel"
    assert segments[0].output_name == "001_Eins.mp3"
    assert segments[1].output_name == "002_Mein_neuer_Titel.mp3"


def test_number_rules_distinguish_year_from_quantity():
    assert numbers_to_words("1987 war gut") == "neunzehnhundertsiebenundachtzig war gut"
    assert numbers_to_words("1987 Menschen") == "tausendneunhundertsiebenundachtzig Menschen"


def test_batch_config_uses_absolute_speaker_and_midpoint():
    directory = Path("generate/audiobook_batch")
    base = directory / ".test-config.json"
    speaker = directory / ".test-voice.pt"
    destination = directory / ".test-session-config.json"
    try:
        base.write_text(json.dumps({"clear_markdown": True, "speaker": "old"}), encoding="utf-8")
        speaker.touch()
        create_batch_config(
            base,
            destination,
            speaker=speaker,
            language="German",
            min_chunk_chars=200,
            max_chunk_chars=500,
        )
        value = json.loads(destination.read_text(encoding="utf-8"))
    finally:
        base.unlink(missing_ok=True)
        speaker.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
    assert value["speaker"] == str(speaker.resolve())
    assert value["target_chunk_chars"] == 350
    assert value["clear_markdown"] is False


def test_extract_multiline_footnote():
    main, footnotes = extract_footnotes("Text[^a]\n\n[^a]: Erste Zeile\n    zweite Zeile")
    assert main.strip() == "Text[^a]"
    assert footnotes == {"a": "Erste Zeile zweite Zeile"}
