# Clean-Markdown-Session

## Zweck

Diese Datei dokumentiert die Arbeiten rund um die TTS-Sprachqualitätstests und
die konfigurierbare Textvorverarbeitung `clear_markdown` in
`generate/generate_mp3_with_embedding.py`.

Die Vorverarbeitung soll für TTS ungeeignete Schreibweisen gezielt ersetzen
oder entfernen. Sie ist in `generate/config.json` standardmäßig aktiviert und
kann pro Aufruf über die CLI ein- oder ausgeschaltet werden. Das Programm
bleibt weiterhin ein einzelnes Python-Script.

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
- `datumszeilen_mit_datum.md`
- `satz_und_absatzpausen.md`
- `satzzeichen_und_intonation.md`
- `sonderzeichen_und_namensaussprache.md`
- `urls_und_technische_schreibweisen.md`
- `woertliche_rede.md`
- `zahlen_und_einheiten.md`
- `zahlen_und_nummerierte_listen.md`

Alle Dateien wurden als UTF-8 eingelesen und ohne Modellinitialisierung mit dem
semantischen Chunker geprüft.

## Konfigurierbarer Standard `clear_markdown`

Die Beispielkonfiguration enthält:

```json
"clear_markdown": true
```

Wenn kein CLI-Override angegeben wird, gilt dieser konfigurierte Wert. Daher
wird die Bereinigung im normalen Aufruf standardmäßig vor dem Chunking
angewendet:

```cmd
python generate\generate_mp3_with_embedding.py ^
  --speaker voices\Sachbuch-Autor ^
  --input samples\tts_sprachqualitaet_tests\datumsangaben.md
```

Die Priorität lautet:

1. `--clear-markdown` beziehungsweise `--clear_markdown` erzwingt die
   Aktivierung.
2. `--no-clear-markdown` beziehungsweise `--no_clear_markdown` erzwingt die
   Deaktivierung.
3. Ohne CLI-Option gilt `clear_markdown` aus der ausgewählten Konfiguration.
4. Fehlt der Schlüssel dort, gilt der eingebaute Standard `true`.

Der effektive Wert wird außerdem als `clear_markdown` im Generation-Report
gespeichert.

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

Markdown-Hervorhebungen dürfen dabei die ganze Zeile, den Bezeichner oder den
Wert umschließen. Unterstützt werden Sternchen, Unterstriche,
Durchstreichungsmarker und Backticks sowie ein vorangestelltes
Überschriften-, Blockzitat- oder Listenpräfix. Beispiele:

```text
_Quelle: https://example.org_
**Quelle: https://example.org**
*Quelle: https://example.org*
**Quelle:** https://example.org
__Quelle__: https://example.org
> **Quelle:** https://example.org
```

Bei einer `Quelle:`-Zeile wird auch die in den Testdaten vorgekommene
fehlerhafte Schreibweise `https:://` tolerant als URL erkannt.

Eine davon getrennte Regel entfernt Zeilen, die außer `Datum:` nur ein gültiges
Datum in deutscher oder ISO-Schreibweise enthalten. Optionaler Leerraum und
ein abschließender Satzpunkt sind erlaubt:

```text
Datum: 01.08.2026
Datum: 2026-08-01
```

Dieselben Markdown-Varianten werden auch für `Datum:` akzeptiert, zum Beispiel
`_Datum: 01.08.2026_`, `**Datum:** 2026-08-01` und
`Datum: **31.02.2026**`.

Für diese Löschregel wird die Kalendergültigkeit bewusst nicht geprüft. Auch
formal passende Zeilen wie `Datum: 31.02.2026` oder `Datum: 99.99.9999` werden
entfernt. Enthält die Zeile weitere Angaben, beispielsweise
`Datum: 01.08.2026, Seite 4`, bleibt sie bestehen; lediglich ein enthaltenes
gültiges Datum wird anschließend wie üblich in Langform umgewandelt.
`Quelle:` gefolgt von einem Datum wird nicht entfernt; für `Quelle:` gilt nur
die URL-Regel.

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
  --print_cleaned_text
```

Alias: `--print-cleaned-text`

stdout enthält ausschließlich den bereinigten Text, sodass die Ausgabe in
eine Pipe oder ein anderes Programm weitergeleitet werden kann.

### Schreiben einer UTF-8-Datei

```cmd
python generate\generate_mp3_with_embedding.py ^
  --input eingabe.md ^
  --write_cleaned_text bereinigt.md
```

Alias: `--write-cleaned-text`

Fehlende Elternverzeichnisse werden angelegt. Die Statusmeldung mit dem
Zielpfad wird auf stderr ausgegeben und verunreinigt daher stdout nicht.

### Beide Ausgaben kombinieren

```cmd
python generate\generate_mp3_with_embedding.py ^
  --input eingabe.md ^
  --print_cleaned_text ^
  --write_cleaned_text bereinigt.md
```

Für beide Ausgabeswitches gilt:

- Der effektive Config-/CLI-Wert von `clear_markdown` muss aktiviert sein.
- Mit dem Standardwert `true` ist kein zusätzlicher Aktivierungsswitch nötig.
- `--speaker` und `--output` sind nicht erforderlich.
- Im Textmodus wird bei fehlendem CLI-Override nur der benötigte
  `clear_markdown`-Wert aus der Konfigurationsdatei gelesen; die vollständige
  TTS-Konfiguration, das Modell und CUDA werden nicht geladen.
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
- Config-Standard sowie positive und negative CLI-Overrides,
- Abkürzungen einschließlich geschützter Leerzeichen,
- Plus-Listenmarker am Zeilenanfang,
- Trennung von Aufzählungsadverbien und Datumsordinalen,
- deutsche und ISO-Datumsangaben,
- gültige Schaltjahresdaten und ungültige Kalenderdaten,
- Entfernung alleinstehender Codeblöcke,
- Erhalt von Inline-Code und unvollständigen Fences,
- Quellenzeilen, nackte URLs, Autolinks und Markdown-Links,
- reine `Datum:`-Zeilen mit formal deutschen oder ISO-Datumsangaben,
- Textausgabe auf stdout,
- UTF-8-Dateiausgabe einschließlich neuem Elternverzeichnis,
- Beenden vor der vollständigen `load_config()`- und `generate_mp3()`-Pipeline,
- Fehler bei deaktiviertem effektivem `clear_markdown` im Textausgabemodus,
- Fehler bei einem vollständig entfernten Eingabetext.

Der zuletzt vollständig ausgeführte Stand war:

```text
88 passed, 3 xfailed, 32 subtests passed
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
- `Quelle:` wird nur als vollständige, am Zeilenanfang stehende URL-Quellenzeile
  speziell behandelt. `Datum:`-Zeilen mit zusätzlichen Angaben werden nicht
  vollständig entfernt.
- Andere URL-Schemata als HTTP, HTTPS und `www.` werden nicht entfernt.
- Die Vorverarbeitung ist auf die vereinbarten deutschen Regeln zugeschnitten.

Diese Punkte sind keine unbemerkten Regressionen, sondern der derzeit bewusst
begrenzte Funktionsumfang.
