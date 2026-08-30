# Clean-Markdown-Session

## Zweck

Diese Datei dokumentiert die Arbeiten rund um die TTS-Sprachqualitätstests und
die optionale Textvorverarbeitung `--clear_markdown` in
`generate/generate_mp3_with_embedding.py`.

Die Vorverarbeitung soll für TTS ungeeignete Schreibweisen gezielt ersetzen
oder entfernen. Ohne den Switch bleibt der bisherige Eingabetext unverändert.
Das Programm bleibt weiterhin ein einzelnes Python-Script.

## Ausgangslage

Für die Beurteilung der deutschen Sprachqualität wurde zunächst eine größere
Markdown-Testdatei erstellt. Sie enthielt unter anderem:

- Aussagen, Fragen, Ausrufe und verschiedene Satzzeichen,
- einzelne Satzzeichen auf eigenen Zeilen,
- nummerierte und nicht nummerierte Aufzählungen,
- Datumsangaben in deutscher und ISO-Schreibweise,
- Satz- und Absatzpausen,
- Markdown-Auszeichnungen und Codeblöcke,
- URLs, technische Schreibweisen und Abkürzungen,
- Zahlen, Einheiten, Sonderzeichen und wörtliche Rede.

Da die Gesamttestdatei für einzelne Hörvergleiche zu lang war, wurden die
Testfälle zusätzlich in kurze, nach ihrem Inhalt benannte Dateien unter
`samples/tts_sprachqualitaet_tests/` aufgeteilt.

Die ursprüngliche Gesamttestdatei
`samples/tts_sprachqualitaet_test.md` blieb als Referenz bestehen.

## Angelegte Sprachqualitätstests

Das Verzeichnis `samples/tts_sprachqualitaet_tests/` enthält:

- `abkuerzungen_und_langtext.md`
- `aufzaehlungen_am_satzanfang.md`
- `aufzaehlungen_im_satz.md`
- `datumsangaben.md`
- `einzelne_satzzeichen.md`
- `gemischter_abschlusstest.md`
- `markdown_codeblock.md`
- `markdown_textformatierungen.md`
- `satz_und_absatzpausen.md`
- `satzzeichen_und_intonation.md`
- `sonderzeichen_und_namensaussprache.md`
- `urls_und_technische_schreibweisen.md`
- `woertliche_rede.md`
- `zahlen_und_einheiten.md`
- `zahlen_und_nummerierte_listen.md`

Alle Dateien wurden als UTF-8 eingelesen und ohne Modellinitialisierung mit dem
semantischen Chunker geprüft.

## CLI-Switch `--clear_markdown`

Der neue Switch wird vor dem Chunking angewendet:

```cmd
python generate\generate_mp3_with_embedding.py ^
  --speaker voices\Sachbuch-Autor ^
  --input samples\tts_sprachqualitaet_tests\datumsangaben.md ^
  --clear_markdown
```

Auch die Schreibweise `--clear-markdown` wird akzeptiert.

Ohne `--clear_markdown` wird keine der nachfolgend beschriebenen
Transformationen ausgeführt.

## Implementierte Transformationen

### Abkürzungen

Folgende Schreibweisen werden ersetzt:

```text
z. B.  -> z.B.
z. B.  -> z.B.           (geschütztes Leerzeichen)
d. h.  -> d.h.
d. h.  -> d.h.           (geschütztes Leerzeichen)
u. a.  -> unter anderem
u. a.  -> unter anderem  (geschütztes Leerzeichen)
u.a.   -> unter anderem
ggf.   -> gegebenenfalls
bzw.   -> beziehungsweise
```

Die Erkennung erfolgt ohne Beachtung der Groß-/Kleinschreibung.

### Markdown-Listenmarker mit Pluszeichen

Ein Pluszeichen wird nur dann in einen Bindestrich umgewandelt, wenn es mit
optionalem Einzug am Zeilenanfang steht und danach Leerraum sowie Text folgen:

```text
+ Eintrag  -> - Eintrag
```

Ein Pluszeichen innerhalb eines Satzes, beispielsweise `1 + 2`, bleibt
unverändert.

### Nummerierte Aufzählungen innerhalb einer Zeile

Nummerierte Marker mitten in einer Zeile werden als deutsche
Aufzählungsadverbien ausgeschrieben:

```text
Ablauf: 1. starten, 2. prüfen, 3. beenden.
Ablauf: erstens starten, zweitens prüfen, drittens beenden.
```

Unterstützt werden die bereits vorhandenen Listenordinale 1 bis 19, 20 bis 99
und 100. Nummerierte Marker am Zeilenanfang werden weiterhin durch die
bestehende Option `speak_numbered_lists` verarbeitet.

Datumsangaben werden vor dieser Listenregel verarbeitet, damit beispielsweise
`01.08.2026` nicht irrtümlich als nummerierte Aufzählung interpretiert wird.

### Datumsangaben

Gültige Kalenderdaten in den folgenden Schreibweisen werden in eine deutsche
Langform umgewandelt:

```text
01.08.2026  -> Erster August 2026
2026-08-01  -> Erster August 2026
02.08.2026  -> Zweiter August 2026
2026-08-03  -> Dritter August 2026
29.02.2024  -> Neunundzwanzigster Februar 2024
```

Für Datumsangaben werden Ordinaladjektive wie `Erster`, `Zweiter` und `Dritter`
verwendet. Die Wörter `erstens`, `zweitens` und `drittens` sind ausschließlich
für Aufzählungen vorgesehen.

Tag und Monat werden mit `datetime.date` geprüft. Ungültige Angaben wie
`31.02.2026` bleiben unverändert. Versionsnummern wie `2.5.1` werden nicht als
Datum behandelt.

Die Umwandlung ist derzeit absichtlich kontextfrei. Deshalb wird immer die
vereinbarte Nominativform ausgegeben. Eine grammatische Anpassung wie
`am ersten August` statt `am Erster August` ist noch nicht implementiert.

### Alleinstehende Codeblöcke

Vollständig geschlossene fenced Codeblöcke mit Backticks oder Tilden werden
entfernt, wenn Öffnungs- und Abschluss-Fence jeweils auf einer eigenen Zeile
stehen:

````text
```python
print("wird entfernt")
```
````

Folgende Inhalte bleiben erhalten:

- Inline-Code wie `` `print` ``,
- Triple-Backticks mitten in einer Textzeile,
- unvollständige beziehungsweise nicht geschlossene Code-Fences.

### Quellenzeilen und URLs

Eine komplette Zeile wird entfernt, wenn sie mit `Quelle:` beginnt und danach
unmittelbar eine HTTP- oder HTTPS-URL enthält:

```text
Quelle: http://example.org
Quelle: https://example.org/artikel
```

Darüber hinaus werden entfernt:

- `http://`-URLs,
- `https://`-URLs,
- `www.`-URLs,
- URL-Ziele in Markdown-Links,
- HTTP-/HTTPS-Autolinks in spitzen Klammern.

Bei einem Markdown-Link bleibt der sichtbare Linktext erhalten:

```text
[Beispielseite](https://example.org) -> Beispielseite
```

E-Mail-Adressen werden nicht als URLs entfernt. Auch Versionsnummern,
IP-Adressen, Dateipfade und Inline-Code bleiben erhalten.

Nach dem Entfernen werden überflüssige Leerzeichen vor Satzzeichen sowie mehr
als zwei aufeinanderfolgende Leerzeilen bereinigt.

## Reine Textausgabe ohne TTS

Zwei weitere Switches erlauben die Kontrolle des bereinigten Textes, ohne die
TTS-Engine zu starten.

### Ausgabe auf stdout

```cmd
python generate\generate_mp3_with_embedding.py ^
  --input eingabe.md ^
  --clear_markdown ^
  --print_cleaned_text
```

Alias: `--print-cleaned-text`

stdout enthält ausschließlich den bereinigten Text, sodass die Ausgabe in
eine Pipe oder ein anderes Programm weitergeleitet werden kann.

### Schreiben einer UTF-8-Datei

```cmd
python generate\generate_mp3_with_embedding.py ^
  --input eingabe.md ^
  --clear_markdown ^
  --write_cleaned_text bereinigt.md
```

Alias: `--write-cleaned-text`

Fehlende Elternverzeichnisse werden angelegt. Die Statusmeldung mit dem
Zielpfad wird auf stderr ausgegeben und verunreinigt daher stdout nicht.

### Beide Ausgaben kombinieren

```cmd
python generate\generate_mp3_with_embedding.py ^
  --input eingabe.md ^
  --clear_markdown ^
  --print_cleaned_text ^
  --write_cleaned_text bereinigt.md
```

Für beide Ausgabeswitches gilt:

- `--clear_markdown` ist erforderlich.
- `--speaker` und `--output` sind nicht erforderlich.
- Konfiguration, TTS-Modell und CUDA werden nicht geladen.
- Nach der Ausgabe beendet sich das Programm erfolgreich.
- Ist der Text nach der Bereinigung leer, endet die CLI mit einem
  verständlichen Eingabefehler.

## Zugehörige Preroll-Fehlerbehebung

Beim ersten Modelllauf mit der großen Sprachqualitätstestdatei brach Chunk 6
mit folgender Meldung ab:

```text
RuntimeError: no suitable text-preroll pause found for chunk 6
```

Die gespeicherte Fehler-WAV enthielt eine gültige Pause, aber mit niedrigem
Restpegel. Mit der normalen Schwelle wurde sie nur für etwa 90 Millisekunden
erkannt. Eine um 5 dB tolerantere Schwelle erkannte denselben Bereich als etwa
170 Millisekunden lange Pause.

Die Pausenerkennung führt deshalb nur nach einem erfolglosen normalen Versuch
einen zweiten Versuch mit einer um 5 dB toleranteren, bei `-40 dBFS`
begrenzten Schwelle aus. Suchfenster, Mindestdauer und Lead-in bleiben
unverändert. Es wurde kein blinder Zeitschnitt eingeführt.

Die reale Fehleraufnahme ergab anschließend:

```text
Pause:   3,450 bis 3,620 Sekunden
Schnitt: 3,590 Sekunden
```

## Tests und Verifikation

Die Tests in `tests/test_generate_mp3_with_embedding.py` prüfen unter anderem:

- beide Schreibweisen aller neuen CLI-Switches,
- Abkürzungen einschließlich geschützter Leerzeichen,
- Plus-Listenmarker am Zeilenanfang,
- Trennung von Aufzählungsadverbien und Datumsordinalen,
- deutsche und ISO-Datumsangaben,
- gültige Schaltjahresdaten und ungültige Kalenderdaten,
- Entfernung alleinstehender Codeblöcke,
- Erhalt von Inline-Code und unvollständigen Fences,
- Quellenzeilen, nackte URLs, Autolinks und Markdown-Links,
- Textausgabe auf stdout,
- UTF-8-Dateiausgabe einschließlich neuem Elternverzeichnis,
- Beenden vor `load_config()` und `generate_mp3()`,
- Fehler bei fehlendem `--clear_markdown`,
- Fehler bei einem vollständig entfernten Eingabetext.

Der zuletzt vollständig ausgeführte Stand war:

```text
82 passed, 3 xfailed, 33 subtests passed
Branch Coverage für generate: 97 %
```

Die drei strikten XFails sind ältere, weiterhin getrennt dokumentierte Fehler:

1. Typprüfung eines geladenen XVector-Prompts erfolgt zu spät.
2. `--characters` greift nicht, wenn der Legacy-Modus nur aus der Konfiguration
   stammt.
3. MP3-Ausgabe wird noch nicht atomar ersetzt.

Nach der expliziten Ergänzung der Datumsfälle `Zweiter` und `Dritter` wurden die
gezielten Datums- und Listenregressionstests erneut erfolgreich ausgeführt.

## Bekannte Abgrenzungen und mögliche Erweiterungen

- Markdown-Fett-, Kursiv- und Durchstreichungsmarker werden derzeit nicht durch
  `--clear_markdown` entfernt.
- Überschriften werden weiterhin erst vom semantischen Chunker behandelt.
- Die Datumsform wird nicht an den grammatischen Kasus des umgebenden Satzes
  angepasst.
- Eine entfernte URL kann einen sprachlich unvollständigen Restsatz erzeugen,
  beispielsweise `Weitere Informationen stehen unter.`
- `Quelle:` wird nur als vollständige, am Zeilenanfang stehende Quellenzeile
  speziell behandelt.
- Andere URL-Schemata als HTTP, HTTPS und `www.` werden nicht entfernt.
- Die Vorverarbeitung ist auf die vereinbarten deutschen Regeln zugeschnitten.

Diese Punkte sind keine unbemerkten Regressionen, sondern der derzeit bewusst
begrenzte Funktionsumfang.
