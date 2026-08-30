# Lange Texte mit `generate_mp3_with_embedding.py` vertonen

Das Skript erzeugt aus einem direkt übergebenen Text oder einer UTF-8-Datei
eine zusammenhängende MP3-Datei mit Qwen3-TTS. Lange Texte werden in sinnvolle
Abschnitte zerlegt, einzeln als Float-PCM erzeugt, nachbearbeitet und am Ende
genau einmal als MP3 codiert.

Die Standardkonfiguration liegt in [`config.json`](config.json). Werte auf der
Kommandozeile überschreiben die entsprechenden Werte aus dieser Datei.

## Voraussetzungen

- eine eingerichtete Python-Umgebung des Projekts
- eine CUDA-fähige NVIDIA-GPU für den normalen schnellen Ausführungspfad
- ein passendes Qwen3-TTS-Modell
- ein mit `examples/extract_speaker.py` und dem passenden Ausgabemodell
  erzeugtes Sprecher-Embedding (`.pt`)
- eine SoundFile/libsndfile-Installation mit MP3-Unterstützung

Die folgenden Beispiele gehen davon aus, dass das aktuelle Verzeichnis das
Repository-Stammverzeichnis `faster-qwen3-tts` ist und die virtuelle Umgebung
bereits aktiviert wurde. Ohne aktivierte Umgebung kann unter Windows
`python` durch `.\.venv\Scripts\python.exe` ersetzt werden.

## Schnellstart

Eine UTF-8- oder UTF-8-BOM-Datei vertonen:

```powershell
python generate\generate_mp3_with_embedding.py --input .\artikel.md
```

Ohne `--output` entsteht neben `artikel.md` automatisch `artikel.mp3`.

Ausgabedatei und Metrikbericht vorgeben:

```powershell
python generate\generate_mp3_with_embedding.py `
  --input .\artikel.md `
  --output .\artikel-hoerbuch.mp3 `
  --metrics .\artikel-hoerbuch.metrics.json
```

Direkt übergebenen Text vertonen; hierbei ist `--output` erforderlich:

```powershell
python generate\generate_mp3_with_embedding.py `
  --text "Das ist ein kurzer Test." `
  --output .\test.mp3
```

Eine andere Konfiguration verwenden:

```powershell
python generate\generate_mp3_with_embedding.py `
  --input .\artikel.md `
  --config .\generate\meine-config.json
```

## Codec-Kontext zwischen Text-Chunks

Das Hauptskript verwendet bei `generation_api: "non_streaming"` standardmäßig
acht Codec-Frames des vorherigen Text-Chunks als Kontext für den kausalen
Audio-Decoder. Der zugehörige Audioanteil wird nach dem Decodieren wieder
abgeschnitten. Dadurch wird der Codec nur beim ersten Chunk kalt gestartet.

Mit `generate_mp3_with_codec_context.py` bleibt zusätzlich das ursprüngliche
Testwerkzeug für direkte A/B-Läufe mit einem per Kommandozeile gewählten Wert
erhalten. Es verwendet ansonsten dieselbe `config.json` und dieselben Optionen:

```powershell
python generate\generate_mp3_with_codec_context.py --input .\artikel.md
```

Ohne `--output` schreibt das Testskript nach
`artikel-codec-context.mp3`, damit die normale Ausgabe nicht überschrieben
wird. Standardmäßig werden 8 Codec-Frames übernommen. Für einen Vergleich
mit kürzerem Kontext kann der Wert geändert werden:

```powershell
python generate\generate_mp3_with_codec_context.py `
  --input .\artikel.md `
  --codec-context-frames 8
```

Der Testpfad benötigt `generation_api: "non_streaming"`, weil nur dort die
Codec-IDs eines vollständigen Text-Chunks vor dem Audio-Decodieren verfügbar
sind. Alle übrigen Werte stammen unverändert aus der gewählten Konfiguration.

## Fester Text-Preroll und sicherer Textauslauf

Das Hauptskript `generate_mp3_with_embedding.py` fängt den hörbaren Modellstart
jedes Text-Chunks standardmäßig durch einen später verworfenen, festen Text ab.
Der Preroll wird standardmäßig bereits vor Chunk 1 gesetzt, damit auch dessen
langsame beziehungsweise klanglich abweichende Anfangsphase verworfen wird.
Der ungefähr vier Sekunden lange Standardsatz endet selbst mit dem Auslaufmarker
`\n\n.`. Er wird einmal allein zur Bestimmung seiner Dauer erzeugt und danach im
selben TTS-Lauf vor jeden eigentlichen Chunk gesetzt. Damit besitzen Kalibrierung
und echter Preroll denselben expliziten Abschluss. Das Skript sucht nahe der
Kalibrierdauer nach der Trennpause und schneidet unmittelbar vor dem folgenden
Sprachbeginn.

Die Standardaufteilung verwendet 220/340/520 Zeichen für Minimum/Ziel/Maximum.
Dabei bevorzugt der semantische Chunker Absatz- und Themengrenzen gegenüber
einer exakt getroffenen Zielgröße. Ein sinnvoll abgeschlossener Absatz darf
daher bis zur Obergrenze von 520 Zeichen wachsen. Nur wenn ein Absatz diese
Grenze überschreitet, wird möglichst an einem vollständigen Satz getrennt. Die
kurzen Chunks sollen verhindern, dass die bei langen Modellläufen beobachtete
Beschleunigung bis zur stark gehetzten Spätphase anwächst.

Unmittelbar vor jeder TTS-Anfrage ergänzt das Hauptskript den Nutztext außerdem
um zwei Zeilenumbrüche und einen einzelnen Punkt (`\n\n.`). Dieser Marker gehört
nicht zum Quelldokument und nicht zur angezeigten Chunklänge. Er gibt dem Modell
aber zusätzlichen Textkontext nach dem letzten gesprochenen Wort, damit das
Chunkende nicht verschluckt wird und vor dem Audioende eine Pause entstehen
kann.

Der normale Aufruf benötigt deshalb keine zusätzlichen Preroll-Parameter:

```powershell
python generate\generate_mp3_with_embedding.py --input .\artikel.md
```

`generate_mp3_with_text_preroll.py` bleibt als A/B-Testwerkzeug erhalten. Damit
lassen sich Prerollsatz, Suchfenster und Chunkgrößen für einzelne Versuche per
Kommandozeile überschreiben:

```powershell
python generate\generate_mp3_with_text_preroll.py --input .\artikel.md
```

Ohne `--output` entsteht `artikel-text-preroll.mp3`. Der normale Inhalt der
`config.json`, einschließlich `codec_context_frames`, wird weiterverwendet.

```powershell
python generate\generate_mp3_with_text_preroll.py `
  --input .\artikel.md `
  --preroll-sentence "Am frühen Morgen lag ein ruhiges Licht über der weiten Landschaft." `
  --preroll-search-window-ms 1200 `
  --preroll-min-pause-ms 120
```

Mit `--preroll-from-chunk 2` lässt sich zum Vergleich das frühere Verhalten
wiederherstellen, bei dem Chunk 1 ohne Preroll erzeugt wird.

Die absatzorientierten Chunkwerte können bei Bedarf angepasst werden:

```powershell
python generate\generate_mp3_with_text_preroll.py `
  --input .\artikel.md `
  --min-chars 220 `
  --target-chars 340 `
  --characters 520 `
  --max-new-tokens 768 `
  --model-max-seq-len 2048
```

Findet das Skript keine ausreichend lange Pause im Suchfenster, bricht es ab
und speichert das ungeschnittene Audio im `_parts`-Verzeichnis. Dadurch wird
kein Zieltext anhand einer bloßen Zeitschätzung angeschnitten.

## Ablauf

1. Das Skript lädt interne Standardwerte und überschreibt sie mit der gewählten
   JSON-Konfiguration.
2. Angegebene CLI-Optionen überschreiben wiederum die Konfiguration.
3. Der Text wird normalisiert und entsprechend `mode` in Chunks aufgeteilt.
4. Modell und Sprecher-Prompt werden einmal geladen.
5. Der feste Preroll wird einmal kalibriert. Danach werden Preroll, Nutztext und
   Auslaufmarker für jeden Chunk in einer TTS-Anfrage erzeugt und der Preroll
   an der erkannten Trennpause entfernt.
6. Die erzeugten Chunks werden nachbearbeitet und als PCM-Abschnitte mit
   passenden Pausen zusammengefügt.
7. Das Gesamtergebnis wird einmal als MP3 codiert.
8. Bei `parts_directory_policy: "delete"` wird das temporäre `_parts`-Verzeichnis
   erst nach erfolgreicher MP3-Codierung gelöscht. Bei einem Fehler bleibt es
   zur Wiederherstellung erhalten.

## Konfigurationsdatei

Die Datei muss ein JSON-Objekt enthalten. Unbekannte Feldnamen werden als
Fehler abgewiesen. Nicht angegebene bekannte Felder erhalten den internen
Standardwert des Skripts. Relative Pfade in der Konfigurationsdatei werden
relativ zum Verzeichnis der Konfigurationsdatei aufgelöst. Ein über
`--speaker` angegebener relativer Pfad wird dagegen relativ zum aktuellen
Arbeitsverzeichnis aufgelöst.

### Modell und Stimme

| Parameter | Wert in `config.json` | Beschreibung |
|---|---:|---|
| `speaker` | `"../Sachbuch-Autor"` | Basisname oder genauer Pfad zum `.pt`-Sprecher-Embedding. Ohne Endung sucht das Skript zuerst `<Name>-1.7B.pt` beziehungsweise `<Name>-0.6B.pt` passend zum Ausgabemodell und danach `<Name>.pt`. Ein gültiges Embedding enthält 2048 Werte. |
| `language` | `"German"` | Sprache für die Synthese, zum Beispiel `German`, `English`, `French`, `Spanish` oder `Auto`. |
| `instruct` | `""` | Optionale Stilbeschreibung. Beim Base-Modell ist sie im X-Vector-Modus experimentell und mit ICL meist verlässlicher. Ein leerer String deaktiviert sie. |
| `model_path` | `"Qwen/Qwen3-TTS-12Hz-1.7B-Base"` | Hugging-Face-Modell-ID oder lokaler Modellpfad. Das Skript ist auf Base Voice Cloning ausgelegt. |
| `device` | `"cuda:0"` | Gerät, auf das das Sprecher-Embedding geladen wird. |
| `model_device` | `"cuda"` | Gerät für das TTS-Modell. Dafür existiert derzeit keine eigene CLI-Option. |
| `dtype` | `"bfloat16"` | Modelldatentyp: `bfloat16`, `float16` oder `float32`. Auf der Titan RTX kann `float16` schneller sein. |

#### Speaker-Embedding mit dem passenden Modell erzeugen

Das Sprecher-Embedding wird mit
[`examples/extract_speaker.py`](../examples/extract_speaker.py) aus einer
Referenzaufnahme erzeugt. Dabei muss exakt dasselbe Modell einschließlich Größe
und Variante verwendet werden, das später auch in `model_path` für die
Sprachausgabe geladen wird. Ein mit dem 0.6B-Modell erzeugtes Embedding darf
nicht für eine 1.7B-Ausgabe verwendet werden und umgekehrt.

Für das 1.7B-Ausgabemodell:

```powershell
python examples\extract_speaker.py `
  --ref_audio .\Veit-Lindau-Referenz.wav `
  --model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base `
  --output .\Veit-Lindau-1.7B.pt
```

Für das 0.6B-Ausgabemodell:

```powershell
python examples\extract_speaker.py `
  --ref_audio .\Veit-Lindau-Referenz.wav `
  --model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base `
  --output .\Veit-Lindau-0.6B.pt
```

Die Größenkennung `-1.7B` beziehungsweise `-0.6B` sollte direkt beim
`--output`-Namen angegeben oder nach der Erzeugung vor `.pt` ergänzt werden.
Wurde zunächst beispielsweise `Veit-Lindau.pt` erzeugt, kann die Datei so
umbenannt werden:

```powershell
Rename-Item .\Veit-Lindau.pt Veit-Lindau-1.7B.pt
```

Sollen beide Modellgrößen verwendet werden, müssen zwei Embeddings mit dem
jeweils passenden Modell erzeugt werden:

```text
Veit-Lindau-0.6B.pt
Veit-Lindau-1.7B.pt
```

In `config.json` bleibt dann nur der gemeinsame Basisname stehen:

```json
{
  "speaker": "../Veit-Lindau",
  "model_path": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
}
```

Das Generate-Skript erkennt `1.7B` im `model_path` und lädt automatisch
`Veit-Lindau-1.7B.pt`. Bei einem 0.6B-Modell wählt es entsprechend
`Veit-Lindau-0.6B.pt`. Die zusätzliche Größenkennung verhindert, dass
versehentlich ein mit der falschen Modellvariante erzeugtes Embedding verwendet
wird. Ein unspezifisches `<Name>.pt` wird nur als Rückfalloption akzeptiert.

### Sampling, Reproduzierbarkeit und Generierung

| Parameter | Wert in `config.json` | Gültige Werte und Wirkung |
|---|---:|---|
| `temperature` | `0.9` | Sampling-Temperatur, mindestens `0`. Höhere Werte erzeugen mehr Variation. |
| `top_k` | `50` | Beschränkt die Auswahl je Schritt auf die wahrscheinlichsten Token. `0` deaktiviert die Begrenzung. |
| `top_p` | `1.0` | Nucleus-Sampling im Bereich `0 < top_p <= 1`. |
| `repetition_penalty` | `1.05` | Wiederholungsstrafe, muss größer als `0` sein. |
| `do_sample` | `true` | `true` verwendet Sampling; `false` wählt deterministischer. |
| `seed` | `1276137643` | Basis-Seed für reproduzierbare Läufe. |
| `seed_strategy` | `"fixed"` | `increment` erhöht den Seed für jeden Chunk; `fixed` verwendet für jeden Chunk denselben Seed. |
| `min_new_tokens` | `2` | Minimale Anzahl neuer Codec-Schritte; Wert mindestens `0`. |
| `warmup_max_new_tokens` | `20` | Maximale Tokenzahl des kurzen Warm-up-Laufs; muss größer als `0` sein. |
| `max_new_tokens` | `768` | Maximale Tokenzahl pro echtem Chunk; muss größer als `0` sein. Der Wert reicht für die kurzen Produktionschunks einschließlich Preroll und Auslaufmarker. |
| `model_max_seq_len` | `2048` | Länge des statischen Talker-KV-Caches. Muss größer als `max_new_tokens` sein und lässt zusätzlichen Platz für den Text-Prefill. |
| `non_streaming_mode` | `true` | Upstream-Option für die Textzuführung im Modell. Sie ist unabhängig von `generation_api`; `false` entspricht beim Voice-Cloning dem schrittweisen Text-Feed. |
| `generation_api` | `"non_streaming"` | `non_streaming` erzeugt einen ganzen Chunk am Stück. `streaming` sammelt die Streaming-Ausgabe wieder zu einer fertigen Datei und dient hier hauptsächlich Vergleichsmessungen. |
| `codec_context_frames` | `8` | Anzahl der Codec-Frames des vorherigen Text-Chunks, die beim Decodieren als akustischer Kontext vorangestellt und anschließend wieder abgeschnitten werden. `8` aktiviert den getesteten Standard; `0` deaktiviert den Cross-Chunk-Kontext für A/B-Vergleiche. Wird nur mit `generation_api: "non_streaming"` angewandt. |

`non_streaming_mode` und `generation_api` klingen ähnlich, steuern aber
verschiedene Dinge: Ersteres bestimmt die Prompt-/Textzuführung innerhalb des
Modells, Letzteres die verwendete Python-Generierungsschnittstelle.

### Chunking und Textvorverarbeitung

| Parameter | Wert in `config.json` | Gültige Werte und Wirkung |
|---|---:|---|
| `mode` | `"semantic"` | `semantic`, `semantic_icl` oder `legacy`; siehe Abschnitt „Betriebsarten“. |
| `legacy_chunk_chars` | `400` | Maximale Zeichen pro Chunk im `legacy`-Modus. Bereich `1` bis `3000`. |
| `target_chunk_chars` | `340` | Angestrebte Chunklänge im semantischen Modus. Bereich `1` bis `3000`. |
| `min_chunk_chars` | `220` | Bevorzugte Mindestlänge semantischer Chunks. Bereich `1` bis `3000`. |
| `max_chunk_chars` | `520` | Harte Maximallänge semantischer Chunks. Vollständige Absätze dürfen zugunsten eines sinnvollen Abschlusses bis zu diesem Wert wachsen. |
| `speak_numbered_lists` | `true` | Wandelt bei `language: "German"` nummerierte Listenmarker am Zeilenanfang um, beispielsweise `1.` in `Erstens,` und `21.` in `Einundzwanzigstens,`. Unterstützt werden Positionen 1 bis 100. Zahlen im Fließtext bleiben unverändert. |
| `append_chunk_end_padding` | `true` | Ergänzt jeden an das Modell gesendeten Chunk um den sicheren Textauslauf `\n\n.`. Quelldokument und Zeichenzählung bleiben unverändert. |
| `chunk_end_padding_text` | `"\n\n."` | Fester Auslaufmarker aus zwei Zeilenumbrüchen und einem Punkt. Bei aktiviertem Auslauf muss dieser Wert exakt so bleiben. |

Für semantische Chunks muss immer gelten:

```text
min_chunk_chars <= target_chunk_chars <= max_chunk_chars
```

Der semantische Modus bevorzugt vollständige Sätze, Absätze, Überschriften und
Themenwechsel. Markdown-Überschriftszeichen wie `#` werden nicht vorgelesen.
Übergroße Einzelsätze werden möglichst an Klausel- oder Wortgrenzen geteilt.

### Pausen, Schnitt und Lautstärke

| Parameter | Wert in `config.json` | Beschreibung |
|---|---:|---|
| `text_preroll_enabled` | `true` | Aktiviert den kalibrierten Text-Preroll vor jedem Chunk einschließlich Chunk 1. Erfordert `generation_api: "non_streaming"`. |
| `text_preroll_sentence` | `"Am frühen Morgen … Landschaft.\n\n."` | Fester, ungefähr vier Sekunden langer Satz einschließlich Auslaufmarker, der vor jedem Nutztext erzeugt und anschließend verworfen wird. |
| `text_preroll_search_window_ms` | `1200` | Suchbereich vor und nach der isoliert gemessenen Prerolldauer. |
| `text_preroll_min_pause_ms` | `120` | Mindestlänge der akzeptierten Trennpause zwischen Preroll und Nutztext. |
| `text_preroll_lead_in_ms` | `30` | Anteil der Trennpause, der vor dem Nutztext erhalten bleibt. |
| `sentence_pause_ms` | `240` | Zielpause an einer normalen Satz-/Chunkgrenze. |
| `paragraph_pause_ms` | `450` | Zielpause nach einem Absatz. |
| `topic_pause_ms` | `650` | Zielpause nach einer Überschrift oder einem erkannten Themenwechsel. |
| `max_leading_silence_ms` | `120` | Maximale Stille, die am Anfang eines erzeugten Chunks erhalten bleibt. |
| `max_trailing_silence_ms` | `160` | Maximale Stille, die am Ende eines erzeugten Chunks erhalten bleibt. |
| `silence_threshold_db` | `-50.0` | Grundschwelle zur Erkennung von Stille. Die tatsächliche Schwelle berücksichtigt zusätzlich den Pegel des Chunks. |
| `edge_fade_ms` | `8` | Kurze Ein-/Ausblendung an Chunkkanten zur Vermeidung von Klicks. |
| `crossfade_ms` | `0` | Überblendungsdauer zwischen Chunks. Eine Überblendung wird nur angewandt, wenn die Zielpause an dieser Grenze `0` ist. Standardmäßig deaktiviert. |
| `loudness_match_max_db` | `1.5` | Maximale Pegelkorrektur eines Chunks relativ zum ersten gesprochenen Chunk; muss mindestens `0` sein. |

Alle Millisekundenwerte müssen mindestens `0` sein. Das Skript misst bereits
vorhandene Randstille und fügt nur den fehlenden Anteil der Zielpause ein.

### WAV-Teile und MP3-Ausgabe

| Parameter | Wert in `config.json` | Beschreibung |
|---|---:|---|
| `save_wav_parts` | `true` | Schreibt jeden fertigen Chunk während der Erzeugung als verlustfreie Float-WAV-Datei nach `<MP3-Name>_parts`. Mit `false` werden keine WAV-Zwischendateien geschrieben. |
| `parts_directory_policy` | `"delete"` | `delete` löscht das exakte `_parts`-Verzeichnis nach erfolgreicher, nicht leerer MP3-Codierung. `keep` behält es. Bei einem Codierungsfehler wird es nicht gelöscht. |

`--no-wav-parts` setzt `save_wav_parts` für den aktuellen Lauf auf `false`.
Das ist nicht dasselbe wie die Standardrichtlinie `delete`: Bei `delete` dienen
die WAV-Dateien während eines langen Laufs als Zwischenstände und werden erst
nach erfolgreichem Abschluss entfernt.

### ICL-Referenz

| Parameter | Wert in `config.json` | Beschreibung |
|---|---:|---|
| `ref_audio` | `""` | Pfad zur Referenzaufnahme für `semantic_icl`. |
| `ref_text` | `""` | Exakte Transkription der Referenzaufnahme. |
| `ref_text_file` | `""` | Alternative zu `ref_text`: UTF-8-Datei mit der exakten Transkription. Ist sie gesetzt, überschreibt ihr Inhalt `ref_text`. |
| `icl_append_silence_ms` | `500` | Stille, die beim Aufbau des ICL-Prompts an die Referenzaufnahme angehängt wird. |

Für `semantic_icl` müssen `ref_audio` und entweder `ref_text` oder
`ref_text_file` vorhanden sein. Die Transkription sollte Wort für Wort zur
Referenzaufnahme passen. Auch die allgemeine `speaker`-Angabe muss nach der
derzeitigen Konfigurationsvalidierung auf eine gültige `.pt`-Datei zeigen.

## Betriebsarten

### `semantic`

Empfohlener Standard für lange deutsche Texte. Verwendet das kompakte
X-Vector-Sprecher-Embedding und verteilt Sätze anhand natürlicher Grenzen auf
Chunks nahe `target_chunk_chars`.

```json
{
  "mode": "semantic"
}
```

### `semantic_icl`

Verwendet denselben semantischen Chunker, baut aber einmalig einen vollständigen
ICL-Prompt aus Referenzaufnahme und exakter Transkription auf.

```json
{
  "mode": "semantic_icl",
  "ref_audio": "../referenz.mp3",
  "ref_text_file": "../referenz.txt"
}
```

Alternativ per CLI:

```powershell
python generate\generate_mp3_with_embedding.py `
  --input .\artikel.md `
  --mode semantic_icl `
  --ref-audio .\referenz.mp3 `
  --ref-text-file .\referenz.txt
```

### `legacy`

Behält den älteren, primär durch eine feste Zeichengrenze gesteuerten Chunker
für Vergleichstests bei. `--sentences` und `--padding` sind ausschließlich in
diesem Modus erlaubt.

```powershell
python generate\generate_mp3_with_embedding.py `
  --input .\artikel.md `
  --mode legacy `
  --characters 900
```

Eine feste Anzahl vollständiger Sätze pro Chunk verwenden:

```powershell
python generate\generate_mp3_with_embedding.py `
  --input .\artikel.md `
  --mode legacy `
  --sentences 4
```

Nur der letzte Chunk darf dabei weniger Sätze enthalten. Überschreitet ein so
gebildeter Chunk das Zeichenlimit, bricht das Skript mit einer verständlichen
Fehlermeldung ab.

## CLI-Optionen

CLI-Werte haben Vorrang vor `config.json`.

| Option | Bedeutung |
|---|---|
| `-h`, `--help` | Zeigt die integrierte Hilfe. |
| `--text TEXT` | Direkt zu vertonender Text. Schließt `--input` aus und erfordert `--output`. |
| `--input PFAD` | UTF-8-Textdatei. Schließt `--text` aus. Ohne `--output` wird die Endung durch `.mp3` ersetzt. |
| `--output PFAD` | Zielpfad; die Endung muss `.mp3` sein. |
| `--config PFAD` | Andere JSON-Konfiguration; Standard ist `generate/config.json`. |
| `--speaker PFAD` | Überschreibt `speaker`; ein relativer Pfad bezieht sich auf das aktuelle Arbeitsverzeichnis. |
| `--language NAME` | Überschreibt `language`. |
| `--model_path ID_ODER_PFAD` | Überschreibt `model_path`. Der Optionsname enthält aus Kompatibilitätsgründen einen Unterstrich. |
| `--device DEVICE` | Überschreibt das Gerät für das Sprecher-Embedding. |
| `--dtype bfloat16\|float16\|float32` | Überschreibt den Modelldatentyp. |
| `--seed ZAHL` | Überschreibt den Basis-Seed. |
| `--mode legacy\|semantic\|semantic_icl` | Überschreibt die Betriebsart. |
| `--generation-api non_streaming\|streaming` | Wählt die Generierungsschnittstelle. |
| `--target-chars N` | Überschreibt `target_chunk_chars`; Bereich 1 bis 3000. |
| `--characters N` | Überschreibt `max_chunk_chars`. Bei Legacy-Aufrufen sollte zusätzlich `--mode legacy` angegeben werden, damit auch `legacy_chunk_chars` gesetzt wird. Bereich 1 bis 3000. |
| `--sentences N` | Exakt N vollständige Sätze pro Chunk; nur mit `legacy`. |
| `--padding` | Hängt an jeden vollständigen Satz zwei Zeilenumbrüche und einen Punkt an; nur mit `legacy`. |
| `--ref-audio PFAD` | Überschreibt `ref_audio` für ICL. |
| `--ref-text TEXT` | Direkte Referenztranskription; schließt `--ref-text-file` aus. |
| `--ref-text-file PFAD` | Datei mit Referenztranskription; schließt `--ref-text` aus. |
| `--instruct TEXT` | Optionale Stilbeschreibung. |
| `--metrics PFAD` | Schreibt einen JSON-Bericht mit Laufzeiten, Chunkdaten, RTF und VRAM-Nutzung. |
| `--no-wav-parts` | Schreibt für diesen Lauf überhaupt keine WAV-Zwischendateien. |

Die vollständige automatisch erzeugte CLI-Hilfe ist jederzeit verfügbar:

```powershell
python generate\generate_mp3_with_embedding.py --help
```

## Ausgaben

### MP3

Die Zieldatei wird erst nach der Synthese aller Chunks geschrieben. Das Skript
verlangt ausdrücklich die Endung `.mp3` und codiert das zusammengefügte Signal
nur einmal.

### Parts-Verzeichnis

Bei `save_wav_parts: true` entstehen während des Laufs Dateien wie:

```text
artikel_parts/
  teil-001.wav
  teil-002.wav
  teil-003.wav
```

Mit der Standardrichtlinie `parts_directory_policy: "delete"` wird dieses
Verzeichnis nach erfolgreicher MP3-Codierung gelöscht. Mit `"keep"` bleibt es
erhalten. Bei einem Fehler vor oder während der MP3-Codierung bleiben bereits
geschriebene Teile ebenfalls erhalten.

### Metrikbericht

`--metrics bericht.json` schreibt unter anderem:

- Modell, Datentyp, Modus und Generierungsschnittstelle
- Textlänge, Chunkanzahl und durchschnittliche Chunklänge
- Gesamtdauer, TTS-Zeit, MP3-Zeit, RTF und Echtzeitfaktor
- Initialisierungs-, Warm-up- und Nachbearbeitungszeit
- maximal belegten und reservierten GPU-Speicher
- pro Chunk Zeichen, Grenze, Seed, Audiodauer, Laufzeit und Decode-Werte

## Hinweise zur Textgestaltung

- Absätze durch Leerzeilen trennen.
- Sätze mit normaler Satzende-Zeichensetzung abschließen.
- Markdown-Überschriften können im semantischen Modus als Themenwechsel dienen.
- Deutsche nummerierte Listen als eigene Zeilen schreiben:

  ```text
  1. Der erste Punkt.
  2. Der zweite Punkt.
  ```

  Mit `speak_numbered_lists: true` wird daraus für die Synthese intern
  „Erstens, … Zweitens, …“. Die Quelldatei wird nicht verändert.
- Sehr lange Sätze ohne Satzzeichen erschweren natürliches Chunking und eine
  gute Betonung.

## Häufige Fehler

- **`speaker embedding not found`:** `speaker` zeigt nicht auf eine vorhandene
  Datei. Bei einem relativen Pfad in der JSON-Datei ist das Verzeichnis der
  JSON-Datei die Basis.

- **`semantic_icl requires both ref_audio and an exact ref_text or
  ref_text_file`:** Für ICL fehlen Referenzaufnahme oder Transkription.

- **`MP3 encoding failed`:** Die installierte SoundFile/libsndfile-Version
  besitzt keine MP3-Unterstützung. Bereits erzeugte WAV-Teile bleiben erhalten.

- **`single sentence ... exceeds the ... character chunk limit`:** Ein
  einzelner Satz beziehungsweise ein per `--sentences` erzwungener Chunk ist
  länger als das konfigurierte Limit. Satz aufteilen oder Limit erhöhen.

- **`unknown config value(s)`:** Die JSON-Datei enthält einen unbekannten oder
  falsch geschriebenen Parameternamen. Das Skript ignoriert solche Fehler
  absichtlich nicht.
