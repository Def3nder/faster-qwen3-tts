# SESSION – Audio-Werkbank für Markdown-zu-MP3-Batches

Stand: 3. Oktober 2026  
Repository: `E:\Code\faster-qwen3-tts`  
Arbeitsbereich: `generate\audiobook_batch`

## 1. Ziel der Arbeit

Für `generate\generate_mp3_with_embedding.py` wurde eine eigenständige Windows-
Oberfläche gebaut. Sie soll lange Markdown-Dokumente sichtbar und kontrollierbar
in einzelne Abschnitte zerlegen und diese Abschnitte nacheinander als MP3-Dateien
erzeugen.

Das wichtigste Anwendungsszenario ist derzeit:

- Quelle: `input\Katrin Hinrichs - Ich frage fur einen Freund.md`
- Teilung: eine MP3 pro Level-3-Überschrift (`###`, also H3)
- Ausgabe: fortlaufend nummerierte Dateien wie `001_Aufschlauen.mp3`
- Verarbeitung: strikt sequenziell
- Kühlung: mindestens 10 Sekunden Pause und, wenn verfügbar, Warten auf eine
  NVIDIA-GPU-Temperatur unter 65 °C

Die Quelldatei wird niemals verändert. Die aufbereitete Vorlesefassung wird nur in
der Vorschau und in temporären Batch-Dateien erzeugt.

## 2. Vorgaben des Nutzers und getroffene Entscheidungen

| Vorgabe | Entscheidung / Umsetzung |
|---|---|
| Moderne, leistungsfähige Oberfläche | PySide6 wurde gewählt. Es bietet native Baum-, Tabellen-, Prozess- und Datei-Dialoge und passt besser als eine einfache Tk-Oberfläche. |
| Hell-/Dunkel-Modus | Oben im Fenster kann zwischen einem vollständigen hellen und dunklen Farbschema gewechselt werden. Die Wahl wird dauerhaft gespeichert. |
| Installation bei Bedarf | PySide6 ist als eigene Abhängigkeit in `requirements.txt` festgehalten und wurde in der vorhandenen, mit `uv venv` erstellten `.venv` installiert. |
| Alles in ein separates Verzeichnis | Die Anwendung liegt in `generate\audiobook_batch`. Nur der zugehörige Repository-Test liegt konventionsgemäß unter `tests`. |
| Markdown per Datei-Dialog öffnen | Unterstützt werden `.md` und `.txt`; gelesen wird UTF-8 beziehungsweise UTF-8 mit BOM. |
| Geöffnete Datei anzeigen | Das Original wird vollständig in einer schreibgeschützten Vorschau angezeigt. |
| Überschriften einklappen | Alle erkannten H1–H6-Überschriften werden hierarchisch in einem Baum dargestellt. |
| Eine Datei pro Überschriften-Level | H1 bis H6 sind auswählbar; H3 ist der Standard. |
| Kapitel ohne Markdown-Überschriften | Der Modus `Sprungmarken aus Inhaltsverzeichnis` liest verlinkte TOC-Ziele und teilt am jeweils passenden HTML-Anker. Bei fehlenden H3-Kapiteln wird er automatisch gewählt, wenn mindestens zwei gültige Ziele erkannt werden. |
| Einleitung separat auswählbar | Text vor der ersten Zielüberschrift wird als `Einleitung` angeboten und ist standardmäßig abgewählt. |
| Feiner schneiden | Der Cursor kann in einen Absatz gesetzt werden; `Schnitt vor Absatz` fügt dort eine Schnittmarke ein. |
| Keine zusätzliche Textbearbeitung | Original- und Vorlesevorschau sind absichtlich schreibgeschützt. Inhalte werden nicht editiert; erlaubt sind Schnittmarken, Dateiauswahl und das Übernehmen einer markierten Textspanne als Dateibereich. |
| Vor dem Start genau sehen, was wohin kommt | Der Dateiplan zeigt Auswahl, laufende Nummer, endgültigen Dateinamen, Zeichenzahl und Status. Die Auswahl einer Zeile markiert den zugehörigen Text. |
| Vorgeschlagene Bereiche anpassen und merken | Für die aktuell gewählte Dateizeile kann eine freie Textmarkierung aus der Originalansicht als individueller Bereich gespeichert werden. Sie wird beim erneuten Anklicken und nach einem Programmneustart wiederhergestellt, solange Pfad und Dokumentinhalt unverändert sind. |
| Fortlaufender Dateiname | Nur ausgewählte Abschnitte werden lückenlos nummeriert: `001_Kapitelname.mp3`, `002_….mp3` usw. |
| MP3-Titel editierbar | Der Titelteil in `Dateiname (editierbar)` kann per Doppelklick geändert werden. Nummer und Endung werden weiterhin automatisch verwaltet; eigene Titel werden dokument- und modusbezogen gespeichert. |
| Sequenzielle Abarbeitung | Es läuft immer genau ein Generatorprozess. Der nächste startet erst nach erfolgreichem Abschluss und der Kühlphase. |
| Dateiplan während des Batchs | Die Liste bleibt scrollbar und Zeilen können zur Textansicht ausgewählt werden. Häkchen, MP3-Titel, Schnittmarken und individuelle Bereiche sind währenddessen gesperrt. |
| Konfigurierbare Wartezeit | Bereich 0 bis 3600 Sekunden, eine Nachkommastelle; Standard 10,0 Sekunden. |
| NVIDIA-Zieltemperatur | Optional per `nvidia-smi`; Standard aktiv, GPU 0, Grenzwert strikt **unter** 65 °C. |
| Temperatur während der Erzeugung | Die Fußzeile fragt die gewählte NVIDIA-GPU ungefähr alle zwei Sekunden ab und zeigt den aktuellen Wert während Generatorlauf und Kühlpause. |
| Keine maximale Wartezeit | Absichtlich umgesetzt. Solange die Temperatur nicht unter dem Grenzwert liegt, wartet die Anwendung unbegrenzt. |
| Stimme konfigurierbar | Auswahl eines vorhandenen `.pt`-Sprecher-Embeddings. |
| Sprache konfigurierbar | Vorbelegt mit `German`; angeboten werden außerdem `English`, `French`, `Spanish`, `Italian` und `Auto`; das Feld ist frei editierbar. |
| Chunk-Größe konfigurierbar | Minimum und Maximum jeweils 50–4000 Zeichen; Standard 220/520, daraus Zielwert 370. |
| Vorleseregeln per Klick | `Vorlesetext aufbereiten` erzeugt die tatsächlich an den Generator übergebene Fassung. `Zurück zum Original` wechselt wieder zur Originalansicht. |
| Aussprachewörterbuch | Semikolon-CSV im Format der vorhandenen Thiaoouba-Datei; `Phonetisch` hat Vorrang vor `Aussprache`. |
| Projektspezifische Ersetzungen | Zweite, separat auswählbare Semikolon-CSV; eine Beispieldatei wurde angelegt. |
| Zahlen und Nummerierungen ausschreiben | Deutsche Kardinalzahlen, Jahreszahlen, Dezimalzahlen und nummerierte Listen werden aufbereitet. Zusammengesetzte Zahlwörter können für Qwen mit Bindestrichen gegliedert werden; Standard aktiv. |
| Fußnoten | Wahlweise vorlesen; Standard aktiv. Eine Fußnote folgt direkt auf den Absatz ihres ersten Verweises. |
| Fußnoten nicht vorlesen | Checkbox kann deaktiviert werden; Verweise und Definitionen verschwinden dann aus dem Vorlesetext. |
| Fehler oder Abbruch | Ein Generatorfehler stoppt den Batch; der Benutzer erhält eine Meldung und Details bleiben im Protokoll. |

## 3. Angelegte und geänderte Dateien

### Anwendung

- `generate\audiobook_batch\app.py`  
  PySide6-Oberfläche, Dateiauswahl, Vorschau, Dateiplan, Prozesssteuerung,
  Kühlphase, Protokoll und Manifest.
- `generate\audiobook_batch\core.py`  
  Qt-unabhängige Kernlogik für Überschriften, Segmente, Dateinamen,
  Fußnoten, Zahlen, Markdown-Bereinigung, CSV-Regeln und Batch-Konfiguration.
- `generate\audiobook_batch\__init__.py`  
  Paketmarkierung.
- `generate\audiobook_batch\README.md`  
  Kurzanleitung für Installation, Start und Bedienung.
- `generate\audiobook_batch\requirements.txt`  
  Enthält `PySide6>=6.8,<7`.
- `generate\audiobook_batch\install_dependencies.cmd`  
  Installiert bevorzugt mit `uv pip` in die vorhandene `.venv`; fällt andernfalls
  auf `pip` zurück.
- `generate\audiobook_batch\start_audiobook_batch.cmd`  
  Startet die Anwendung mit `.venv\Scripts\python.exe`, sofern vorhanden.
- `generate\audiobook_batch\project_replacements.example.csv`  
  Beispiel für projektspezifische Ersetzungen.
- `generate\audiobook_batch\aussprache.csv`  
  Für beide analysierten Beispielbücher kuratiertes Aussprachewörterbuch mit 144
  Einträgen.
- `generate\audiobook_batch\assets\audio_werkbank.png`
  Transparente, hochauflösende Fassung des gewählten Icons mit Kapitelstapel,
  Play-Symbol und Audiowelle.
- `generate\audiobook_batch\assets\audio_werkbank.ico`
  Windows-Icon mit Größen von 16 bis 256 Pixeln. Die Anwendung setzt zusätzlich
  eine eigene Windows-App-ID, damit nicht das Python-Standardicon verwendet wird.

### Tests

- `tests\test_audiobook_batch_core.py`  
  Fünfzehn Tests der Qt-unabhängigen Kernlogik.

### Dokumentation

- `SESSION.md`  
  Diese Übergabe- und Entscheidungsdokumentation.

Alle genannten Anwendungs- und Testdateien sind zum aktuellen Stand noch nicht in
Git eingecheckt (`git status` zeigt sie als untracked).

## 4. Oberfläche und Bedienablauf

1. `generate\audiobook_batch\start_audiobook_batch.cmd` starten.
2. Über `Markdown öffnen` die Quelldatei wählen.
3. Im Dokumentbereich das gewünschte Teilungs-Level wählen; für die Beispieldatei
   H3 verwenden.
4. Optional den Cursor in einen Absatz setzen und `Schnitt vor Absatz` wählen.
5. Im Dateiplan Einleitung, Zwischentexte oder Kapitel an- beziehungsweise
   abwählen.
   Der Titelteil eines Dateinamens kann in der Spalte `Dateiname (editierbar)` per
   Doppelklick geändert werden. Leeren des Feldes stellt den automatisch aus dem
   Kapiteltitel erzeugten Namen wieder her.
6. Falls der vorgeschlagene Bereich einer Datei nicht passt: Dateizeile wählen,
   im Tab `Original` den gewünschten Text exakt markieren und
   `Markierung übernehmen` anklicken. `Vorschlag wiederherstellen` entfernt die
   individuelle Anpassung.
7. Stimme, Sprache, Chunk-Minimum, Chunk-Maximum und Ausgabeordner einstellen.
8. Optional `aussprache.csv` und eine Projektregel-CSV auswählen.
9. Fußnoten- und Zahlwortoptionen einstellen.
10. `Vorlesetext aufbereiten` anklicken und Original/Vorlesetext vergleichen.
11. Pause, Temperaturgrenze und GPU-Index kontrollieren.
12. `Batch starten`.

Wenn nach einer Regeländerung die Vorschau veraltet ist, zeigt die Oberfläche
`REGELN GEÄNDERT · NEU AUFBEREITEN`. Beim Batch-Start wird eine fehlende oder
veraltete Vorlesefassung automatisch neu erzeugt.

Beim Anklicken einer Dateizeile bleibt der vollständige zugehörige Bereich
markiert, die Ansicht scrollt aber an den Anfang der Markierung. Das gilt sowohl
für `Original` als auch für den bereits erzeugten `Vorlesetext`.

## 5. Aufteilung des Markdown-Dokuments

### Erkennung

- Erkannt werden ATX-Überschriften `#` bis `######`.
- Überschriften innerhalb von mit Backticks oder Tilden eingezäunten Codeblöcken
  werden ignoriert.
- Der Überschriftentext wird für Baum und Dateiname von einfacher Inline-
  Markdown-Auszeichnung bereinigt.
- Setext-Überschriften mit `===` oder `---` werden bei der Teilung nicht als
  Überschrift verwendet.

Zusätzlich gibt es die inhaltsverzeichnisbasierte Erkennung:

- Verwendet werden Listeneinträge wie `[Kapitelname](#a-4)`.
- Im Dokument muss ein passender HTML-Anker wie `<a id="a-4"></a>` oder
  `<a name="a-4"></a>` vorhanden sein.
- Nur Ziele, die sowohl im Inhaltsverzeichnis als auch im Dokument vorkommen,
  werden übernommen.
- Doppelte Ziele werden nur einmal verwendet.
- Die Kapitelreihenfolge folgt der Position der Zielanker im Dokument.
- Der sichtbare Linktext aus dem Inhaltsverzeichnis wird zum Kapiteltitel.

### Segmentregeln

- Text vor der ersten Überschrift des gewählten Levels wird `Einleitung`.
- Die Einleitung ist standardmäßig nicht ausgewählt.
- Jede Zielüberschrift eröffnet ein Kapitel.
- Eine nachfolgende Überschrift gleicher oder höherer Hierarchie beendet das
  Kapitel.
- Verbleibender Text außerhalb der Zielkapitel wird als `Zwischentext – …`
  angeboten und ist standardmäßig abgewählt.
- Gibt es keine Überschrift des gewählten Levels, wird das gesamte Dokument als
  abgewählte Einleitung angeboten. Der Benutzer muss dann bewusst auswählen oder
  ein anderes Level wählen.
- Im Sprungmarken-Modus wird Text vor der ersten verlinkten Marke ebenfalls als
  separat abgewählte Einleitung behandelt.
- Manuelle Schnitte rasten am Anfang des aktuellen Absatzes ein.
- Ein geschnittenes Kapitel wird als ursprünglicher Titel und anschließend als
  `Titel – Teil 2`, `Titel – Teil 3` usw. dargestellt.
- Ein individuell markierter Bereich ersetzt für genau diese Dateizeile den
  vorgeschlagenen Start und das vorgeschlagene Ende. Titel und Dateiname bleiben
  erhalten.
- Individuelle Bereiche dürfen frei gewählt werden. Dadurch sind bewusst auch
  Lücken oder Überschneidungen mit Nachbarbereichen möglich; der tatsächlich
  markierte Text bleibt im Dateiplan und in der Vorschau kontrollierbar.
- Eine leere oder ausschließlich aus Leerraum bestehende Markierung wird nicht
  akzeptiert.

### Dateinamen

- Nur ausgewählte Segmente erhalten eine Nummer.
- Die Mindestbreite beträgt drei Stellen.
- Für Windows unzulässige Zeichen werden entfernt beziehungsweise durch
  Unterstriche ersetzt.
- Leerzeichen und Punkte werden zu Unterstrichen normalisiert.
- Deutsche Umlaute werden ausgeschrieben (`Ä/Ö/Ü` → `Ae/Oe/Ue`,
  `ä/ö/ü` → `ae/oe/ue`); `ß` wird zu `ss` und `ẞ` zu `SS`.
- Der bereinigte Titel wird auf 120 Zeichen begrenzt.
- Nicht ausgewählte Zeilen zeigen als Dateiname `—`.
- Die Dateinamensspalte ist für ausgewählte Zeilen editierbar.
- Beim Speichern eines eigenen Namens werden eine eingegebene führende Nummer und
  die `.mp3`-Endung entfernt; anschließend setzt die Anwendung die jeweils
  aktuelle fortlaufende Nummer und Endung wieder ein.
- Eigene Titel bleiben erhalten, wenn die Segmentauswahl und damit die laufende
  Nummer geändert wird.
- Leeren des Feldes entfernt die Anpassung und stellt den automatischen Titel
  wieder her.

## 6. Erkenntnisse zur Beispieldatei

Datei:
`input\Katrin Hinrichs - Ich frage fur einen Freund.md`

Aktuell gemessene Eigenschaften:

- 356.972 Zeichen
- 1.788 Zeilen
- 110 Überschriften insgesamt
- 74 H3-Überschriften
- Bei H3-Teilung entstehen 77 Segmente:
  - 1 Einleitung
  - 74 standardmäßig ausgewählte Kapitel
  - 2 standardmäßig abgewählte Zwischentexte

Damit entspricht H3 der gewünschten Regel „eine Audiodatei je Level-3-
Überschrift“.

Zweite analysierte Datei:
`input\Stefan Hiene - Aufwachmedizin - Dein radikaler Weg zur Selbstannahme.md`

- 198.966 Zeichen
- 3.793 Zeilen
- keine H3-Kapitelstruktur
- 57 verlinkte Inhaltsverzeichnis-Ziele
- 58 erzeugte Segmente:
  - 1 standardmäßig abgewählte Einleitung mit Titel, Impressum, Widmung und
    Inhaltsverzeichnis
  - 57 standardmäßig ausgewählte Dateien
- Die 57 Dateien umfassen Dosierungsanleitung, Packungsbeilage, Vorwort, 52
  nummerierte Kapitel, `Über die Aufwachmedizin` und `Über den Autor`.
- Alle 57 Linkziele besitzen einen passenden Anker; es fehlt kein Ziel.
- Die Anwendung wählt für diese Datei automatisch
  `Sprungmarken aus Inhaltsverzeichnis`.

## 7. Aufbereitung des Vorlesetextes

Die Vorschau und die tatsächlich übergebenen Segmentdateien verwenden dieselbe
Kernfunktion. Dadurch soll kein unsichtbarer Unterschied zwischen Vorschau und
Generierung entstehen.

Die Reihenfolge ist absichtlich festgelegt:

1. Projektspezifische wörtliche Ersetzungen anwenden.
2. Allgemeine Abkürzungen ersetzen:
   `z. B.`, `z.B.`, `d. h.`, `d.h.`, `u. a.`, `u.a.`, `ggf.`, `bzw.`, `usw.`
3. Nummerierte Listen als Ordnungswörter formulieren, zum Beispiel
   `1.` → `Erstens,`.
4. Markdown in lesbaren Klartext umwandeln. HTML-Kommentare, Skript-/Style-Blöcke,
   reine HTML-Tags und Markdown-Bilder werden entfernt; normaler Linktext und
   Text innerhalb von Format-Tags bleiben erhalten.
5. ` = ` als ` gleich ` und `OK` als `okay` formulieren.
6. Zahlen ausschreiben.
7. Wiederholte Punkte und Fragezeichen normalisieren.
8. Leerzeichen vor Satzzeichen und mehrfache Leerzeichen bereinigen.
9. Optional zusammengesetzte Zahlwörter mit Bindestrichen gliedern.
10. Aussprachewörterbuch anwenden.

### Zahlen

- Deutsche Kardinalzahlen werden bis in große Größenordnungen erzeugt.
- Dezimalzahlen mit Komma werden gesprochen; Nachkommastellen werden
  ziffernweise formuliert.
- Tausenderpunkte beziehungsweise Tausenderleerzeichen werden entfernt.
- Vierstellige Werte von 1100 bis 2099 werden normalerweise als Jahreszahl
  behandelt.
- Vor den Mengenwörtern `Jahre`, `Jahren`, `Tonnen`, `Menschen`, `Kinder`,
  `Kilometer`, `Meter` und `Personen` werden sie als normale Menge behandelt.
- Listen-Ordnungswörter sind direkt für 1–20 und 100 sowie zusammengesetzt für
  21–99 vorgesehen.

### Fußnoten

- Markdown-Fußnotendefinitionen werden gesammelt und aus dem normalen Text
  entfernt.
- Beim ersten Verweis in einem Absatz wird anschließend
  `Anmerkung. <Fußnotentext>` eingefügt.
- Mehrzeilige, eingerückte Fußnoten werden zusammengeführt.
- Innerhalb einer MP3 wird dieselbe Fußnote nur einmal gesprochen.
- Jede MP3 bleibt absichtlich eigenständig: Verweist ein späteres Kapitel erneut
  auf dieselbe Fußnote, wird sie dort wieder einmal gesprochen.
- Bei deaktivierter Option werden weder Verweis noch Fußnotentext gesprochen.

## 8. CSV-Regeln

Alle CSV-Dateien sind UTF-8 beziehungsweise UTF-8 mit BOM und verwenden ein
Semikolon als Trennzeichen.

### Aussprachewörterbuch

Format:

```text
Schreibweise;Aussprache;Phonetisch;Hinweis
Thiaoouba;Tiauba;Ti-a-u-ba;Beispiel
```

Regeln:

- Pflichtspalten: `Schreibweise`, `Aussprache`, `Phonetisch`.
- `Hinweis` ist fachlich vorgesehen und sollte mitgeführt werden.
- Ist `Phonetisch` gefüllt, wird dieser Wert benutzt.
- Sonst wird `Aussprache` benutzt.
- Leere, wirkungslose oder identische Einträge werden nicht geladen.
- Längere Schreibweisen werden zuerst ersetzt.
- Die Ersetzung ist groß-/kleinschreibungssensitiv und vermeidet Treffer mitten
  in normalen Wörtern.

Die erweiterte `aussprache.csv` enthält 144 eindeutige Einträge. Alle
Schreibweisen kommen in mindestens einer der beiden analysierten Markdown-Dateien
vor; es gibt keine Dubletten und keinen leeren oder wirkungslosen Zielwert.
Abgedeckt sind unter anderem:

- Abkürzungen wie `GANPF`, `AFE`, `MDMA`, `INXS`, `ISBN` und `ZDF`
- englische Begriffe und Marken wie `Dirty Talk`, `Slow Sex`, `WhatsApp`,
  `Womanizer` und `Youporn`
- medizinische beziehungsweise fremdsprachige Begriffe wie `Pedicatio`,
  `Sildenafil`, `Cantharidin`, `Yohimbin` und `Yoni`
- Namen und Buchtitel aus Text und Literaturverzeichnis
- die zweite Datei betreffende Verlagsnamen, sichtbare Web-/E-Mail-Adressen,
  englische Liedzeilen und Wendungen sowie fremdsprachige Personen- und Ortsnamen

Nicht sicher belegte Namensaussprache wurde in `Hinweis` als `Vorschlag`
markiert. Diese Einträge benötigen eine Hörprobe; besonders zu prüfen sind
`Michael Sztenc`, `Femtasy`, `Jesper Bay-Hansen`, `Katherine Woodward Thomas`,
`Love Base Media`, `It Started With a Kiss`, `Loukotka`, `Lars Muhl`,
`Immaculee Ilibagiza` und der englische AFE-Langtext.

### Projektspezifische Ersetzungen

Format:

```text
Suche;Ersatz;Hinweis
Aremo X3;Aremo X drei;Produktname
```

Diese Regeln sind einfache, wörtliche und groß-/kleinschreibungssensitive
Ersetzungen. Sie werden vor der allgemeinen Bereinigung und vor dem
Aussprachewörterbuch angewendet. Ein leerer Ersatz kann Text gezielt entfernen.

## 9. Generatoraufruf und Batch-Zustand

Die vorhandene Datei `generate\generate_mp3_with_embedding.py` wurde nicht
umgebaut. Die Oberfläche startet sie je ausgewähltem Segment mit demselben
Python-Interpreter, mit dem die Oberfläche läuft:

```text
python -u generate_mp3_with_embedding.py
  --config <Ausgabe>\.audiobook_batch\config.json
  --input  <Ausgabe>\.audiobook_batch\segment_XXXX.md
  --output <Ausgabe>\NNN_Kapitelname.mp3
```

Die tatsächliche Befehlszeile wird als Argumentliste ohne Shell-Zwischenschritt
ausgeführt. Standardausgabe und Fehlerausgabe werden gemeinsam live im Protokoll
angezeigt.

### Abgeleitete Konfiguration

Aus `generate\config.json` wird für jeden Batch eine Kopie erstellt. Überschrieben
werden ausschließlich:

- absoluter Pfad zum Sprecher-Embedding
- Sprache
- minimale Chunk-Größe
- maximale Chunk-Größe
- Ziel-Chunk-Größe als arithmetische Mitte
- `clear_markdown = false`, weil die Oberfläche bereits den geprüften Klartext
  übergibt

Alle anderen Generatorparameter bleiben aus `generate\config.json` erhalten.

### Laufdaten

Im Ausgabeordner entsteht `.audiobook_batch` mit:

- `config.json` – Batch-spezifische Generatorkonfiguration
- `segment_XXXX.md` – exakt vorbereiteter Text je ausgewähltem Segment
- `batch_plan.json` – Quelle, Aktualisierungszeit, Split-Level, manuelle Schnitte,
  Titel, Ausgabedateien, Auswahl, Status sowie Start, Ende und Kennzeichnung eines
  individuellen Bereichs; der Teilungsmodus und individuell gesetzte
  Dateinamentitel werden ebenfalls gekennzeichnet

### Fortsetzen und Überspringen

- Existiert die Ziel-MP3 bereits, wird sie als `Übersprungen` markiert.
- Dadurch kann ein abgebrochener Batch grundsätzlich erneut gestartet werden.
- Es gibt derzeit keinen Inhalts-Hash und keine Prüfung, ob eine vorhandene MP3
  noch zur aktuellen Quelle oder zu den aktuellen Regeln passt.
- Soll eine Datei nach Text- oder Regeländerungen neu erzeugt werden, muss die
  vorhandene MP3 vorher bewusst entfernt oder umbenannt werden.

### Fehlerbehandlung

- Rückgabecode 0: Segment wird `Fertig`.
- Rückgabecode ungleich 0: Segment wird `Fehler (<Code>)`, Manifest wird
  aktualisiert, der gesamte Batch hält an.
- Ein Benutzerabbruch beendet den laufenden Prozess sofort und leert die weitere
  Warteschlange.
- Eine automatische Wiederholung fehlgeschlagener Dateien gibt es nicht.

## 10. Kühl- und Temperatursteuerung

Nach jeder erfolgreich erzeugten Datei, außer nach der letzten, beginnt die
Kühlphase.

- Die Mindestpause läuft immer vollständig ab.
- Während des gesamten Batchlaufs wird die Temperatur ungefähr alle zwei Sekunden
  in der Fußzeile aktualisiert, auch während der Generatorprozess läuft.
- Während der Kühlphase zeigt der Batch-Status nur die verbleibende Mindestpause
  in ganzen, aufgerundeten Sekunden beziehungsweise das Warten auf den Grenzwert.
  Die aktuelle Temperatur steht ausschließlich im separaten GPU-Feld und wird
  nicht doppelt angezeigt.
- Bei aktivierter Temperaturprüfung wird ungefähr alle zwei Sekunden abgefragt:

```text
nvidia-smi --id=<GPU> --query-gpu=temperature.gpu --format=csv,noheader,nounits
```

- Weiter geht es nur, wenn Mindestpause erfüllt **und** Temperatur kleiner als
  der Grenzwert ist.
- Gleichheit reicht nicht: Bei einem Grenzwert von 65 °C startet der nächste Job
  erst bei 64 °C oder weniger.
- Es gibt absichtlich kein Zeitlimit.
- Die Abfrage selbst hat ein Timeout von drei Sekunden.
- Ist `nvidia-smi` nicht vorhanden, liefert einen Fehler oder keinen lesbaren
  Wert, protokolliert die Anwendung dies einmal und verwendet nur die
  Mindestpause.
- Unter Windows wird `nvidia-smi` ohne sichtbares Konsolenfenster gestartet.

## 11. Gespeicherte Einstellungen

Über `QSettings("Qwen3-TTS", "AudiobookBatch")` werden gespeichert:

- Sprecherpfad
- Sprache
- Chunk-Minimum und -Maximum
- Aussprache-CSV
- Projektregel-CSV
- Mindestpause
- Temperaturgrenze
- GPU-Index
- zuletzt geöffnete Quelle
- individuelle Textbereiche, getrennt nach Quelldatei und Überschriften-Level
- individuelle MP3-Titel, getrennt nach Quelldatei und Teilungsmodus
- gewählter Hell-/Dunkel-Modus

Die zuletzt geöffnete Quelle wird derzeit nur gespeichert, aber nicht automatisch
wieder geöffnet.

Derzeit **nicht** dauerhaft gespeichert werden:

- Ausgabeordner
- Teilungs-Level
- manuelle Schnittmarken
- Segmentauswahl
- Fußnoten-Checkbox
- Zahlwort-Bindestrich-Checkbox
- Aktivierung der GPU-Temperaturprüfung

Diese Werte beginnen bei jedem Programmstart wieder mit ihren Standardwerten.
Manuelle Schnitte stehen zwar im Manifest, werden daraus aber noch nicht wieder in
die Oberfläche geladen. Individuelle Textbereiche werden dagegen unmittelbar in
`QSettings` gespeichert. Ein SHA-256-Fingerabdruck des vollständigen Dokuments
verhindert, dass alte Zeichenpositionen nach einer Inhaltsänderung auf den falschen
Text angewendet werden.

Ist `voices\Sachbuch-Autor-1.7B.pt` vorhanden, wird es als Standardstimme gewählt;
andernfalls das alphabetisch erste `.pt`-Embedding im Ordner `voices`.

Beim ersten Öffnen einer Quelle wird als Ausgabeordner automatisch
`<Quellordner>\audio_<Dateiname-ohne-Endung>` vorgeschlagen, sofern noch kein
Ausgabeordner eingetragen ist.

## 12. Installation und Start

Die Python-Umgebung wurde mit `uv venv` erstellt. Erwarteter Interpreter:

```text
E:\Code\faster-qwen3-tts\.venv\Scripts\python.exe
```

Einmalige beziehungsweise reproduzierbare Installation:

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r .\generate\audiobook_batch\requirements.txt
```

Start:

```powershell
.\.venv\Scripts\python.exe .\generate\audiobook_batch\app.py
```

Zusätzliche Entwickleroptionen:

```powershell
# Datei beim Start öffnen
.\.venv\Scripts\python.exe .\generate\audiobook_batch\app.py --open .\input\datei.md

# Oberfläche für eine visuelle Kontrolle fotografieren und beenden
.\.venv\Scripts\python.exe .\generate\audiobook_batch\app.py --screenshot .\pfad\ui.png
```

## 13. Validierung und Teststand

Am 3. Oktober 2026 erneut ausgeführt:

```text
.venv\Scripts\python.exe -m pytest tests/test_audiobook_batch_core.py -q
15 passed in 0.07s
```

Die Warnung beim Lauf betraf ausschließlich fehlende Berechtigung zum Schreiben
des Pytest-Caches `.pytest_cache`; die Tests selbst waren erfolgreich.

Abgedeckt sind:

- H3-Aufteilung und separat abgewählte Einleitung
- manuelle Schnitte am Absatzanfang
- Übernahme einer frei markierten Textspanne als Segmentbereich
- Ablehnung einer leeren individuellen Markierung
- Entfernung von HTML-Ankern, Kommentaren, Format-Tags und Markdown-Bildern
- Kapitelaufteilung über Inhaltsverzeichnislinks und HTML-Sprungmarken
- editierter MP3-Titel mit weiterhin automatischer Nummer und Endung
- Fußnote nach erstem Absatzverweis
- deaktivierte Fußnoten
- eigenständige Fußnoten je Ausgabedatei
- Zahlen, Listen, Projektregeln und phonetisches Wörterbuch
- lückenlose Dateinummerierung nur ausgewählter Segmente
- Unterschied zwischen Jahreszahl und Mengenangabe
- abgeleitete Batch-Konfiguration
- mehrzeilige Fußnoten

Ein früherer vollständiger Repository-Testlauf nach Erstellung der Anwendung
ergab 100 bestandene Tests, 3 erwartete XFAIL-Fälle und 42 Subtests. Nach späteren
Nach Änderungen an der Audio-Werkbank wurde zuletzt der gezielte Satz der 15 Core-Tests
erneut ausgeführt. Ein danach angestoßener vollständiger Lauf wurde in einem
bestehenden, ausgabefreien Langzeit-Modelltest abgebrochen; bis dahin war kein
Fehler ausgegeben worden.

Die Aussprache-CSV wurde zusätzlich validiert:

- 144 geladene Einträge
- alle vier erwarteten Spalten vorhanden
- keine doppelten Schreibweisen
- kein leerer oder wirkungsloser Aussprachewert
- jede Schreibweise kommt in mindestens einer der beiden Beispiel-Markdown-Dateien vor

Noch nicht als End-to-End-Test ausgeführt wurde ein kompletter 74-Dateien-
Audiobatch mit realem Modell und GPU.

## 14. Gelöste Erkenntnis: HTML- und Bildmarkierungen im Vorlesetext

Ein dokumentweiter Test mit der echten Beispieldatei hatte zunächst gezeigt:

- 67 rohe HTML-Anker in den 74 standardmäßig ausgewählten H3-Kapiteln;
- 220 aus Markdown-Bildern übernommene Alt-Texte `image`;
- mehrzeilige Hervorhebungszeichen in der Einleitung.

Diese Inhalte wären Teil der Segmentdateien geworden und hätten vom Sprachmodell
interpretiert oder ausgesprochen werden können. Die Bereinigung wurde deshalb in
`core.py` erweitert:

- HTML-Kommentare sowie `script`- und `style`-Blöcke werden entfernt.
- HTML-Tags einschließlich `<a id="…"></a>` werden entfernt, ihr normaler
  Textinhalt bleibt erhalten.
- Markdown-Bilder werden vollständig entfernt; weder Alt-Text noch Bildpfad wird
  gesprochen.
- Normale Markdown-Links behalten ihren sichtbaren Linktext.
- Mehrzeilige Fett- und Kursivmarkierungen werden bereinigt.

Ergebnis der erneuten Prüfung aller 77 H3-basierten Segmente der Beispieldatei:

- 0 HTML-Anker
- 0 erkannte HTML-Tags
- 0 Markdown-Bildmarkierungen
- 0 doppelte Sternchen aus Fettmarkierungen

Vier Vorkommen des Wortes `Image` bleiben absichtlich bestehen. Sie stammen aus
regulärem deutschen Fließtext wie „kein gutes Image“ oder „Imageproblem“ und
sind keine Bildplatzhalter.

## 15. Weitere bekannte Grenzen und bewusste Nicht-Ziele

- Die Oberfläche ist ein lokales Desktop-Werkzeug, kein Webdienst.
- Es gibt keine Textbearbeitung in der Oberfläche; das ist eine bewusste Vorgabe.
- Die automatisch vorgeschlagenen MP3-Dateinamen sind in der Segmentliste direkt
  editierbar.
- Neben ATX-Überschriften können Kapitel über die Sprungmarken eines verlinkten
  Inhaltsverzeichnisses erkannt werden.
- Projektregeln und Ausspracheeinträge sind absichtlich exakt und
  groß-/kleinschreibungssensitiv.
- Ein vorhandener Ausgabedateiname gilt ohne Inhaltsprüfung als abgeschlossen.
- Batch-Manifeste werden geschrieben, aber noch nicht als vollständige Sitzung
  wieder eingelesen. Individuelle Bereiche werden unabhängig davon über
  `QSettings` wiederhergestellt.
- Die lokale `aussprache.csv` wird nicht automatisch vorausgewählt. Sie muss beim
  ersten Mal über die Oberfläche gewählt werden; danach merkt sich `QSettings`
  den Pfad.
- Namen mit als `Vorschlag` markierter Aussprache benötigen Hörproben.
- Die GPU-Temperatursteuerung unterstützt NVIDIA über `nvidia-smi`; für AMD oder
  andere Sensorquellen gibt es keine Implementierung.
- Bei dauerhaft zu hoher Temperatur wartet der Batch unbegrenzt. Das ist
  ausdrücklich gewünscht, muss beim unbeaufsichtigten Betrieb aber bedacht
  werden.

## 16. Empfohlene nächsten Schritte

Priorität 1:

1. `generate\audiobook_batch\aussprache.csv` in der Oberfläche auswählen.
2. Einen kurzen Testlauf mit einem einzigen, repräsentativen Kapitel erzeugen.
3. Aussprache, Pausen, Zahlen und Fußnoten anhören und die als Vorschlag
   markierten Einträge korrigieren.
4. Danach erst den vollständigen 74-Dateien-Batch starten.

Optional:

5. Lokale `aussprache.csv` automatisch als Standard vorbelegen.
6. Manifest-Wiederaufnahme mit Wiederherstellung von Schnitten und Auswahl bauen.
7. Vorhandene MP3-Dateien über Quell-/Regel-Hashes statt nur über ihre Existenz
    beurteilen.
8. Weitere UI-Optionen dauerhaft speichern.

## 17. Entscheidungen, die bei späteren Änderungen erhalten bleiben sollten

- Das Originaldokument bleibt unverändert.
- Die angezeigte Vorlesefassung ist exakt die Fassung, die an den Generator geht.
- Die Qt-unabhängige Textlogik bleibt getrennt von der Oberfläche und direkt
  testbar.
- Einleitung und Zwischentexte dürfen nicht stillschweigend verschwinden; sie
  bleiben separat sichtbar und auswählbar.
- Standardteilung für das Beispiel bleibt H3.
- Nummerierung richtet sich nur nach ausgewählten Dateien und bleibt lückenlos.
- Erzeugung bleibt sequenziell.
- Mindestpause und Temperaturbedingung müssen beide erfüllt sein.
- Der Temperaturvergleich bleibt „kleiner als“, nicht „kleiner oder gleich“.
- Es gibt kein implizites maximales Temperatur-Wartezeitlimit.
- Fußnoten stehen direkt nach dem Absatz ihres ersten Verweises und jede MP3 ist
  in Bezug auf Fußnoten eigenständig.
- Der vorhandene MP3-Generator bleibt eine separat gestartete Komponente; seine
  übrigen Konfigurationswerte werden nicht unnötig von der Oberfläche dupliziert.
