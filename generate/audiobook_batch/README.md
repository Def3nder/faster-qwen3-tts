# Audio-Werkbank

Die Audio-Werkbank zerlegt eine Markdown-Datei anhand ihrer Überschriften und ruft
`../generate_mp3_with_embedding.py` für jede ausgewählte Ausgabedatei einzeln auf.
Die Quelldatei wird niemals verändert.

## Installation und Start

1. Einmal `install_dependencies.cmd` ausführen.
2. Danach `start_audiobook_batch.cmd` doppelklicken.
3. Über **Markdown öffnen** eine UTF-8-Datei auswählen.

Mit **Heller Modus** beziehungsweise **Dunkler Modus** lässt sich die Darstellung
oben im Fenster umschalten. Die Auswahl bleibt für den nächsten Start gespeichert.
Fenster und Windows-Taskleiste verwenden das eigene Audio-Werkbank-Icon aus dem
Unterordner `assets`.

Alternativ in PowerShell:

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r .\generate\audiobook_batch\requirements.txt
.\.venv\Scripts\python.exe .\generate\audiobook_batch\app.py
```

## Arbeitsablauf

1. Überschriften-Level auswählen, zum Beispiel `H3`.
   Bei Büchern mit verlinkten Kapitelmarken kann stattdessen
   **Sprungmarken aus Inhaltsverzeichnis** verwendet werden. Fehlen H3-Kapitel und sind passende
   Sprungmarken vorhanden, wählt die Anwendung diesen Modus automatisch.
2. Optional im Originaltext den Cursor in einen Absatz setzen und **Schnitt vor Absatz** wählen.
3. Zum Anpassen eines vorgeschlagenen Bereichs die Dateizeile wählen, im Original den gewünschten
   Text markieren und **Markierung übernehmen** anklicken. **Vorschlag wiederherstellen** macht die
   Anpassung rückgängig.
4. Ein Sprecher-Embedding, Sprache sowie minimale und maximale Chunk-Größe festlegen.
5. Optional ein Aussprachewörterbuch und projektspezifische Ersetzungen auswählen.
6. **Vorlesetext aufbereiten** wählen und Original/Vorlesetext vergleichen.
7. Im Dateiplan Einleitung oder Kapitel an- und abwählen.
   Der Titelteil eines MP3-Dateinamens lässt sich per Doppelklick auf
   **Dateiname (editierbar)** ändern. Nummer und `.mp3`-Endung bleiben automatisch korrekt.
   Deutsche Umlaute und `ß` werden dabei dateisystemfreundlich als `Ae`, `Oe`,
   `Ue`, `ae`, `oe`, `ue` beziehungsweise `ss` ausgeschrieben.
8. **Batch starten**.

Individuelle Bereiche werden für die Kombination aus Quelldatei, Dokumentinhalt und
Überschriften-Level gespeichert. Beim späteren Öffnen werden sie wieder markiert. Ändert sich der
Inhalt der Quelldatei, werden die alten Positionen aus Sicherheitsgründen nicht übernommen.

Der Vorlesetext enthält keine HTML-Tags oder Markdown-Bilder. Anker wie
`<a id="a-vier"></a>`, Bildpfade und generische Bildtexte wie `image` werden nicht an das
Sprachmodell übergeben. Normaler Linktext und Text innerhalb von Format-Tags bleiben erhalten.

Vorhandene MP3-Dateien werden übersprungen. Dadurch kann ein abgebrochener Lauf
fortgesetzt werden. Laufdaten und temporäre Segmenttexte liegen im gewählten
Ausgabeordner unter `.audiobook_batch`.

## CSV-Formate

Das Aussprachewörterbuch nutzt das Format:

```text
Schreibweise;Aussprache;Phonetisch;Hinweis
Thiaoouba;Tiauba;Ti-a-u-ba;Beispiel
```

`Phonetisch` wird bevorzugt; ist es leer, wird `Aussprache` verwendet.

Projektspezifische Ersetzungen nutzen:

```text
Suche;Ersatz;Hinweis
Aremo X3;Aremo X drei;Produktname
```

Ein vollständiges Beispiel liegt in `project_replacements.example.csv`.

## Temperatursteuerung

Nach jeder erzeugten Datei wartet die Anwendung mindestens die konfigurierte Zeit.
Ist die Temperatursteuerung aktiv, beginnt die nächste Datei zusätzlich erst, wenn
`nvidia-smi` für die gewählte GPU einen Wert **unter** dem Grenzwert meldet. Es gibt
keine maximale Wartezeit. Ist `nvidia-smi` nicht verfügbar, gilt nur die Mindestpause.
Während des gesamten Batchlaufs zeigt die Fußzeile die aktuelle GPU-Temperatur an und
aktualisiert sie ungefähr alle zwei Sekunden – auch während eine MP3 erzeugt wird.
