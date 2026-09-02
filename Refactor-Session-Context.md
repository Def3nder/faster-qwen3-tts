# Refactor-Session-Context

## Zweck dieses Dokuments

Dieses Dokument hält den Stand der Refactoring-Session für
`generate/generate_mp3_with_embedding.py` fest. Es soll als Übergabekontext für
weitere Arbeiten dienen und insbesondere beschreiben:

- welche funktionalen Verträge vor dem Refactoring abgesichert wurden,
- welche Tests ergänzt und wie sie ausgeführt wurden,
- welche Refactorings bereits erfolgt sind,
- welche bekannten Fehler bewusst noch nicht behoben wurden,
- welche offenen Arbeiten als Nächstes sinnvoll sind.

Die zentrale Vorgabe war, das Programm weiterhin als **ein einzelnes Script**
zu behalten. Es wurde daher nicht in weitere Python-Module aufgeteilt.

## Betroffene Dateien

- `generate/generate_mp3_with_embedding.py`
  - produktives Script und Gegenstand des Refactorings
- `tests/test_generate_mp3_with_embedding.py`
  - schnelle, weitgehend isolierte Regressionstests
- `test_generate_mp3_with_embedding.cmd`
  - Windows-Aufruf für die gezielte Testsuite einschließlich Coverage
- `pyproject.toml`
  - pytest-, Coverage- und optionale Test-Abhängigkeiten
- `setup.sh`
  - Installation der Test-Abhängigkeiten unter Linux/macOS
- `setup_windows.bat`
  - Installation der Test-Abhängigkeiten unter Windows
- `.gitignore`
  - Ausschluss von pytest- und Coverage-Artefakten

## Ausgangslage und Refactoring-Ziele

Das Script war vor dem Refactoring sehr lang und enthielt in
`generate_mp3()` einen großen Anteil der gesamten Orchestrierung. Dort waren
unter anderem Modellinitialisierung, Chunk-Generierung, Sample-Rate-Prüfung,
Nachbearbeitung, Metriken, MP3-Ausgabe und Abschlussbericht miteinander
verzahnt.

Die wesentlichen Ziele waren:

1. Funktionales Verhalten vor strukturellen Änderungen durch Tests festhalten.
2. Große Verantwortungsblöcke in kleine interne Funktionen zerlegen, ohne das
   Script in Module aufzuteilen.
3. Datenflüsse durch Typen und kleine Dataclasses verständlicher machen.
4. Wiederholte Logik zentralisieren.
5. Die Audiozusammenführung effizienter machen, ohne die erzeugten PCM-Daten zu
   verändern.
6. Nach jedem größeren Schritt die Regressionstests ausführen.

## Test-Infrastruktur

pytest wurde als optionales Test-Extra in `pyproject.toml` ergänzt:

- `pytest`
- `pytest-cov`

Zusätzlich enthält `pyproject.toml` die pytest- und Coverage-Konfiguration:

- Testverzeichnis: `tests`
- pytest-Ausgabe: `-ra`
- Branch Coverage ist aktiviert.
- Coverage-Quelle ist das Package `generate`.

Sowohl `setup.sh` als auch `setup_windows.bat` installieren das Projekt mit dem
Test-Extra. Das geschieht auch dann, wenn die virtuelle Umgebung bereits
existiert, damit eine ältere Umgebung die neu hinzugekommenen Testwerkzeuge
nachinstalliert bekommt.

pytest wurde in der lokalen `.venv` erfolgreich installiert und verwendet.

## Tests ausführen

Unter Windows ist der bevorzugte Aufruf aus dem Repository-Hauptverzeichnis:

```cmd
test_generate_mp3_with_embedding.cmd
```

Das Script prüft zunächst, ob `.venv\Scripts\python.exe` existiert, führt dann
die gezielte Testsuite mit Branch Coverage aus und gibt den ursprünglichen
pytest-Exitcode zurück.

Der entsprechende direkte Aufruf lautet:

```cmd
.venv\Scripts\python.exe -m pytest -q tests\test_generate_mp3_with_embedding.py --cov=generate --cov-branch --cov-report=term-missing
```

## Abgesicherter Teststand

Der zuletzt verifizierte Stand der gezielten Testsuite war:

```text
88 passed, 3 xfailed, 32 subtests passed
Branch Coverage für generate: 97 %
```

Zusätzlich wurden erfolgreich ausgeführt:

- Python-Syntaxprüfung mit `py_compile`
- Prüfung auf Whitespace- und Patch-Probleme mit `git diff --check`

Die Tests decken unter anderem folgende Bereiche ab:

- Laden und Validieren der Konfiguration
- Auflösung relativer Konfigurationspfade
- Kommandozeilen-Parser und `main()`
- semantisches und altes zeichenbasiertes Chunking
- ein Golden-Resultat des semantischen Chunkings
- Audiozusammenführung mit exakten PCM-Erwartungen
- Pausen, Crossfades und DSP-Nachbearbeitung
- Streaming- und Non-Streaming-Generierung
- ICL- und XVector-Prompts
- Preroll-Verhalten
- Sample-Rate-Konsistenz
- Seed-Strategien
- Metriken und Generation-Report
- Dateiausgabe und relevante Fehlerpfade

Die Tests mocken Modell und Audio-Encoder dort, wo echte Modellgewichte,
CUDA oder FFmpeg für die geprüfte Logik nicht erforderlich sind. Dadurch sind
sie schnell und deterministisch genug, um nach jedem Refactoring-Schritt
ausgeführt zu werden.

## Bekannte Fehler als strikte XFails

Drei bereits vorhandene Fehlverhalten sind als `xfail(strict=True)`
dokumentiert. Sie wurden im Refactoring bewusst nicht nebenbei behoben, damit
strukturelle Änderungen und Verhaltensänderungen getrennt bleiben.

### 1. Nicht-Tensor beim Laden eines XVector-Prompts

Test:
`test_load_xvector_prompt_reports_non_tensor_as_value_error`

Aktuelles Verhalten: `load_xvector_prompt()` ruft `.to()` auf dem geladenen
Objekt auf, bevor geprüft wurde, ob es tatsächlich ein Tensor ist. Bei einem
ungeeigneten Objekt entsteht deshalb nicht der vorgesehene `ValueError`.

### 2. `--characters` bei Legacy-Modus aus der Konfiguration

Test:
`test_main_applies_characters_to_legacy_mode_selected_by_config`

Aktuelles Verhalten: `--characters` wird nicht korrekt angewendet, wenn der
Legacy-Chunking-Modus ausschließlich über die Konfigurationsdatei gewählt
wurde.

### 3. Bestehende MP3-Datei bei fehlgeschlagener Codierung

Test:
`test_write_mp3_preserves_existing_output_when_encoding_fails`

Aktuelles Verhalten: Die MP3-Ausgabe wird direkt in die endgültige Zieldatei
geschrieben. Schlägt die Codierung fehl, kann dadurch eine bereits vorhandene
Ausgabedatei beschädigt oder überschrieben werden. Eine atomare Ausgabe über
eine temporäre Datei mit anschließendem Ersetzen fehlt noch.

Da diese Tests strikt sind, führt auch ein unerwartetes Bestehen (`XPASS`) zum
Fehlschlag der Testsuite. Wird einer der Fehler bewusst behoben, muss der
zugehörige Test in einen normalen Regressionstest umgewandelt werden.

## Durchgeführte Refactorings

### Typisierte Verträge

Für zentrale Werte wurden Type Aliases und strukturierte Typen ergänzt, unter
anderem für:

- Chunk-Grenzen
- Chunking-Modus
- Generation-API
- Seed-Strategie
- Policy des Parts-Verzeichnisses
- DType-Namen
- Konfigurationsdaten (`TTSConfig`)
- Chunk-Timing
- Codec-Zustand
- Chunk-Metriken
- Generation-Report

Auch die Boundary-Felder von `TextChunk` und `_SemanticUnit` sind nun explizit
typisiert. `load_config()` liefert den typisierten Konfigurationsvertrag.

Diese Typisierung ändert das Laufzeitverhalten nicht, macht aber die erlaubten
Werte und die Datenstruktur für weitere Refactorings nachvollziehbarer.

### Effizientere Audiozusammenführung

`join_audio_chunks()` hat vorher bei normalen Chunk-Grenzen wiederholt große
Arrays zusammengefügt. Das erzeugt bei langen Texten unnötige Kopien und kann
quadratisches Laufzeit- und Speicherverhalten begünstigen.

Jetzt werden die Segmente zunächst gesammelt und für den normalen Pfad nur
einmal zusammengefügt. Die bereits vorhandene Stille am Ende wird inkrementell
verfolgt. Nur der Crossfade-Pfad materialisiert bei Bedarf den bis dahin
angesammelten Audioteil.

Die Tests vergleichen relevante Ergebnisse exakt auf PCM-Ebene. Sie blieben
nach dieser Änderung unverändert grün.

### Zentralisierte Sampling-Argumente

Die gemeinsamen Sampling-Parameter der verschiedenen Generierungswege werden
nun über `_sampling_kwargs()` aufgebaut. Dadurch sind Streaming- und
Non-Streaming-Pfade weniger anfällig für auseinanderlaufende Parameter.

### Zerlegung der Orchestrierung innerhalb desselben Scripts

`generate_mp3()` wurde in kleinere, interne Verantwortungsblöcke zerlegt. Das
Script bleibt dabei eine Datei. Zu den neuen Strukturen und Hilfsfunktionen
gehören:

- `_GenerationRuntime`
- `_GenerationState`
- `_PostprocessedChunk`
- `_FinalizedAudio`
- `_normalize_text_chunks()`
- `_initialize_generation_runtime()`
- `_build_chunk_generator()`
- `_run_chunk_generator()`
- `_accept_sample_rate()`
- `_postprocess_chunk()`
- `_build_chunk_metric()`
- `_finalize_audio()`
- `_build_generation_report()`
- `_print_generation_summary()`

`generate_mp3()` koordiniert diese Bausteine jetzt hauptsächlich, statt alle
Details selbst zu enthalten. Die extern sichtbaren Schnittstellen und
Ausgabeformate wurden dabei beibehalten.

### Robustere Erkennung der Text-Preroll-Pause

Ein realer Durchlauf mit dem Sprachqualitätstest scheiterte in Chunk 6, obwohl
zwischen Text-Preroll und Nutztext eine Pause vorhanden war. Die gespeicherte
Fehler-WAV zeigte dort leises Restaudio: Mit der normalen Schwelle von
`-50 dBFS` wurde die Pause nicht für die verlangten 120 Millisekunden erkannt,
mit `-45 dBFS` dagegen als Pause von etwa 170 Millisekunden.

Die Erkennung führt deshalb nur nach einem erfolglosen normalen Versuch einen
zweiten Versuch mit einer um 5 dB toleranteren, bei `-40 dBFS` begrenzten
Schwelle durch. Suchfenster, Mindestdauer und Lead-in bleiben unverändert. Ein
blinder Zeitschnitt oder die Auswahl einer weiter entfernten Pause wurde nicht
eingeführt. Ein neuer Regressionstest bildet eine solche Pause mit niedrigem
Restpegel nach.

### Konfigurierbare Standard-Textbereinigung `clear_markdown`

Der Konfigurationswert `clear_markdown` bereinigt den Eingabetext vor dem
Chunking und ist in `generate/config.json` sowie im eingebauten Standard auf
`true` gesetzt. `--clear-markdown` erzwingt die Aktivierung;
`--no-clear-markdown` deaktiviert sie für einen einzelnen Lauf. Die
Bereinigung umfasst:

- die vereinbarten deutschen Abkürzungen,
- Pluszeichen als Markdown-Listenmarker am Zeilenanfang,
- nummerierte Aufzählungen innerhalb einer Zeile,
- gültige deutsche und ISO-Datumsangaben in deutscher Langform,
- alleinstehende fenced Codeblöcke,
- `Quelle:`-Zeilen mit HTTP-/HTTPS-URL,
- `Datum:`-Zeilen mit einem alleinstehenden formal passenden Datum, ohne
  Prüfung der Kalendergültigkeit,
- Markdown-Hervorhebungen um `Quelle:`, `Datum:` oder deren vollständige Zeile,
- Markdown-Weblinks, HTTP-/HTTPS-URLs und `www.`-URLs.

Inline-Code und unvollständige Code-Fences bleiben erhalten. Bei
Markdown-Links bleibt der sichtbare Linktext stehen. Ungültige Kalenderdaten
werden nicht verändert. Wird der gesamte Inhalt durch die Bereinigung
entfernt, beendet sich die CLI mit einem verständlichen Eingabefehler.

Mit `--print_cleaned_text` kann der bereinigte Text auf stdout ausgegeben
werden. `--write_cleaned_text PATH` schreibt denselben Text als UTF-8-Datei.
Beide Optionen setzen einen effektiv aktivierten Config-/CLI-Wert voraus,
können kombiniert werden und beenden das Programm vor dem Laden der
vollständigen TTS-Konfiguration, des TTS-Modells oder von CUDA.
Die Statusmeldung der Dateiausgabe geht an stderr, sodass stdout für Pipes nur
den bereinigten Text enthält. Auch die Schreibweisen `--print-cleaned-text`
und `--write-cleaned-text` werden akzeptiert.

## Bewusst beibehaltene funktionale Verträge

Folgende Eigenschaften sollten bei weiteren Refactorings nicht unbeabsichtigt
geändert werden:

- Namen und Bedeutung der CLI-Argumente
- flache JSON-Struktur der Konfigurationsdatei
- Vorrangregeln zwischen CLI und Konfiguration
- semantische und Legacy-Chunking-Ergebnisse
- Reihenfolge, Pausen und Crossfades der Audio-Chunks
- Streaming- und Non-Streaming-Aufrufe des Modells
- ICL- und XVector-Verhalten
- Sample-Rate-Prüfungen
- Seed-Verhalten
- Benennung und Inhalt der Parts-Dateien
- Schlüssel und Bedeutung des Generation-Reports
- Exitcodes und relevante Fehlermeldungen

Laufzeitwerte und Timing-Metriken sind naturgemäß nicht bitgenau
reproduzierbar. Die Tests prüfen deshalb deren Struktur und Plausibilität,
nicht identische Zeitmessungen.

## Grenzen der bisherigen Verifikation

Die gezielte Testsuite bietet eine starke Absicherung der Python-Logik und der
extern beobachtbaren Verträge, ersetzt aber keinen vollständigen Test mit einem
echten Modell.

Ein repositoryweiter pytest-Lauf wurde begonnen, sammelt jedoch auch
`tests/test_e2e_parity.py`. Auf einem CUDA-fähigen System kann dieser Test echte
Modelle laden und erhebliche Zeit sowie Ressourcen beanspruchen. Dieser Lauf
wurde deshalb nicht als Teil der schnellen Refactoring-Schleife abgeschlossen.

Noch separat zu prüfen sind daher bei Bedarf:

- reale Modellinferenz mit den vorgesehenen Modellgewichten
- CUDA-spezifisches Verhalten
- tatsächliche FFmpeg/MP3-Codierung in der Zielumgebung
- subjektive beziehungsweise wahrnehmungsbasierte Audioqualität
- Performance und Speicherverbrauch bei sehr langen Eingabetexten

## Empfohlene nächste Schritte

1. Die drei strikten XFails einzeln beheben. Nach jedem Fix den Marker entfernen
   und den Test als normalen Regressionstest bestehen lassen.
2. Die MP3-Ausgabe atomar machen: in eine temporäre Datei im Zielverzeichnis
   schreiben und erst nach erfolgreicher Codierung ersetzen.
3. Besitz und Lebenszyklus des Parts-Verzeichnisses weiter absichern, damit
   fremde oder alte Dateien nicht versehentlich als aktuelle Ausgabe behandelt
   oder gelöscht werden.
4. Den teuren E2E-/CUDA-Test mit einem pytest-Marker klar von der schnellen
   Unit-/Regressionstestsuite trennen und separat ausführen.
5. Optional `mypy` oder `pyright` ergänzen, um die neu eingeführten Typverträge
   automatisiert zu nutzen. Das ist für die Laufzeitfunktion nicht erforderlich.
6. `join_audio_chunks()` mit einem langen realistischen Text benchmarken, um die
   erwartete Verbesserung bei Laufzeit und maximalem Speicherverbrauch zu
   quantifizieren.

## Arbeitsregel für weitere Refactorings

Für jeden größeren Schritt sollte weiterhin gelten:

1. Nur eine zusammengehörige Strukturänderung durchführen.
2. `test_generate_mp3_with_embedding.cmd` ausführen.
3. Prüfen, dass weiterhin `88 passed`, `3 xfailed` und die Subtests erfolgreich
   gemeldet werden, solange keine bewusste Fehlerbehebung erfolgt ist.
4. Bei verändertem Verhalten zuerst klären, ob es eine Regression oder eine
   gewollte Vertragsänderung ist.
5. Gewollte Änderungen mit einem eigenen Test und einer Aktualisierung dieses
   Dokuments festhalten.
