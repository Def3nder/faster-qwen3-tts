#!/usr/bin/env python3
"""PySide6 desktop UI for turning Markdown sections into an MP3 batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from PySide6.QtCore import QProcess, QSettings, Qt, QTimer, Signal
    from PySide6.QtGui import QColor, QFont, QFontDatabase, QTextCharFormat, QTextCursor
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as error:
    if error.name == "PySide6":
        raise SystemExit(
            "PySide6 fehlt. Bitte zuerst install_dependencies.cmd aus diesem Ordner starten."
        ) from error
    raise

try:
    from .core import (
        SpeechOptions,
        Segment,
        assign_output_names,
        build_segments,
        create_batch_config,
        load_pronunciation_csv,
        load_project_replacements,
        paragraph_start,
        parse_headings,
        prepare_segments_for_speech,
        read_utf8_text,
        set_segment_range,
    )
except ImportError:
    from core import (  # type: ignore[no-redef]
        SpeechOptions,
        Segment,
        assign_output_names,
        build_segments,
        create_batch_config,
        load_pronunciation_csv,
        load_project_replacements,
        paragraph_start,
        parse_headings,
        prepare_segments_for_speech,
        read_utf8_text,
        set_segment_range,
    )


APP_DIR = Path(__file__).resolve().parent
GENERATE_DIR = APP_DIR.parent
REPO_ROOT = GENERATE_DIR.parent
GENERATOR_SCRIPT = GENERATE_DIR / "generate_mp3_with_embedding.py"
BASE_CONFIG = GENERATE_DIR / "config.json"
DEFAULT_INPUT_DIR = REPO_ROOT / "input"
DEFAULT_VOICE_DIR = REPO_ROOT / "voices"


DARK_STYLE = """
QWidget {
    background: #111619;
    color: #e8e4dc;
    font-family: "Segoe UI";
    font-size: 10.5pt;
}
QMainWindow { background: #0b0f11; }
QFrame#header {
    background: #171e21;
    border-bottom: 1px solid #344047;
}
QLabel#eyebrow {
    color: #e8a84e;
    font-family: "Bahnschrift SemiCondensed";
    font-size: 9pt;
    letter-spacing: 2px;
}
QLabel#title {
    color: #fffaf0;
    font-family: "Bahnschrift SemiCondensed";
    font-size: 25pt;
    font-weight: 600;
}
QLabel#subtitle { color: #8ea0a8; font-size: 10pt; }
QLabel#sectionTitle {
    color: #f2c879;
    font-family: "Bahnschrift SemiCondensed";
    font-size: 12pt;
    font-weight: 600;
}
QLabel#muted, QLabel.muted { color: #819097; }
QLabel#temperature { color: #73d6cb; font-family: "Cascadia Mono"; }
QPushButton {
    background: #202a2e;
    border: 1px solid #3a484f;
    border-radius: 5px;
    color: #eee9df;
    padding: 7px 13px;
}
QPushButton:hover { background: #2a363b; border-color: #6c7d84; }
QPushButton:pressed { background: #151c1f; }
QPushButton:disabled { color: #59656a; border-color: #283136; }
QPushButton#accent {
    background: #e8a84e;
    border-color: #f4c879;
    color: #17120b;
    font-weight: 700;
}
QPushButton#accent:hover { background: #f0b85f; }
QPushButton#danger { border-color: #9d5147; color: #f2aaa0; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #0e1315;
    border: 1px solid #354249;
    border-radius: 4px;
    padding: 6px 8px;
    selection-background-color: #9e6e31;
}
QComboBox::drop-down { border: none; width: 24px; }
QTreeWidget, QTableWidget, QPlainTextEdit {
    background: #0d1214;
    alternate-background-color: #12191c;
    border: 1px solid #2f3a40;
    border-radius: 5px;
    selection-background-color: #364e55;
    selection-color: #ffffff;
    gridline-color: #263136;
}
QTreeWidget::item { padding: 5px 2px; }
QTreeWidget::item:selected { border-left: 3px solid #e8a84e; }
QHeaderView::section {
    background: #1a2327;
    color: #aebbc0;
    border: none;
    border-right: 1px solid #313c42;
    border-bottom: 1px solid #313c42;
    padding: 7px;
    font-weight: 600;
}
QTabWidget::pane { border: 1px solid #303b40; border-radius: 4px; }
QTabBar::tab {
    background: #151c1f;
    color: #8f9da3;
    border: 1px solid #2d383d;
    padding: 8px 15px;
}
QTabBar::tab:selected { color: #f5c26d; background: #20292d; }
QGroupBox {
    border: 1px solid #303b40;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 600;
    color: #d9b36f;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
QProgressBar {
    background: #0b1012;
    border: 1px solid #354147;
    border-radius: 4px;
    color: #e9e5dc;
    text-align: center;
    min-height: 18px;
}
QProgressBar::chunk { background: #45a99d; border-radius: 3px; }
QSplitter::handle { background: #20292d; width: 2px; height: 2px; }
QScrollBar:vertical { background: #0d1214; width: 11px; }
QScrollBar::handle:vertical { background: #3c4a50; border-radius: 5px; min-height: 30px; }
QToolTip { background: #f2e7d4; color: #17120b; border: 1px solid #e8a84e; }
"""

LIGHT_STYLE = """
QWidget {
    background: #f4f1ea;
    color: #20282c;
    font-family: "Segoe UI";
    font-size: 10.5pt;
}
QMainWindow { background: #e9e5dc; }
QFrame#header {
    background: #fffdf8;
    border-bottom: 1px solid #c9c4b9;
}
QLabel#eyebrow {
    color: #9a5d08;
    font-family: "Bahnschrift SemiCondensed";
    font-size: 9pt;
    letter-spacing: 2px;
}
QLabel#title {
    color: #172125;
    font-family: "Bahnschrift SemiCondensed";
    font-size: 25pt;
    font-weight: 600;
}
QLabel#subtitle { color: #66757b; font-size: 10pt; }
QLabel#sectionTitle {
    color: #86530b;
    font-family: "Bahnschrift SemiCondensed";
    font-size: 12pt;
    font-weight: 600;
}
QLabel#muted, QLabel.muted { color: #6b777c; }
QLabel#temperature { color: #087f75; font-family: "Cascadia Mono"; }
QPushButton {
    background: #fffdf8;
    border: 1px solid #b9b7b0;
    border-radius: 5px;
    color: #263136;
    padding: 7px 13px;
}
QPushButton:hover { background: #eee9df; border-color: #888d8d; }
QPushButton:pressed { background: #dfdbd2; }
QPushButton:disabled { color: #a4a7a5; border-color: #d5d1c9; }
QPushButton#accent {
    background: #d98c20;
    border-color: #b76c08;
    color: #1d160c;
    font-weight: 700;
}
QPushButton#accent:hover { background: #e9a33f; }
QPushButton#danger { border-color: #a54e42; color: #8d3027; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #fffefa;
    border: 1px solid #bbb8b0;
    border-radius: 4px;
    padding: 6px 8px;
    selection-background-color: #e8b76e;
    selection-color: #1d2528;
}
QTreeWidget, QTableWidget, QPlainTextEdit {
    background: #fffefa;
    alternate-background-color: #f2eee6;
    border: 1px solid #c4c0b7;
    border-radius: 5px;
    selection-background-color: #b8d9d4;
    selection-color: #142225;
    gridline-color: #d4d0c7;
}
QTreeWidget::item { padding: 5px 2px; }
QTreeWidget::item:selected { border-left: 3px solid #c9790f; }
QHeaderView::section {
    background: #e7e2d9;
    color: #445158;
    border: none;
    border-right: 1px solid #c8c3ba;
    border-bottom: 1px solid #c8c3ba;
    padding: 7px;
    font-weight: 600;
}
QTabWidget::pane { border: 1px solid #c5c0b7; border-radius: 4px; }
QTabBar::tab {
    background: #e9e5dc;
    color: #627076;
    border: 1px solid #c8c3ba;
    padding: 8px 15px;
}
QTabBar::tab:selected { color: #7b4c08; background: #fffdf8; }
QGroupBox {
    border: 1px solid #c5c0b7;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 600;
    color: #77501a;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
QProgressBar {
    background: #fffefa;
    border: 1px solid #bbb8b0;
    border-radius: 4px;
    color: #293337;
    text-align: center;
    min-height: 18px;
}
QProgressBar::chunk { background: #63b9ad; border-radius: 3px; }
QSplitter::handle { background: #c9c5bc; width: 2px; height: 2px; }
QScrollBar:vertical { background: #eeeae2; width: 11px; }
QScrollBar::handle:vertical { background: #a9aaa6; border-radius: 5px; min-height: 30px; }
QToolTip { background: #252d30; color: #fffaf0; border: 1px solid #976015; }
"""


class PathPicker(QWidget):
    changed = Signal(str)

    def __init__(self, mode: str = "file", filter_text: str = "Alle Dateien (*)") -> None:
        super().__init__()
        self.mode = mode
        self.filter_text = filter_text
        self.edit = QLineEdit()
        self.edit.textChanged.connect(self.changed)
        button = QPushButton("Auswählen")
        button.clicked.connect(self._browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.edit, 1)
        layout.addWidget(button)

    def _browse(self) -> None:
        current = self.edit.text().strip()
        start = current or str(REPO_ROOT)
        if self.mode == "directory":
            value = QFileDialog.getExistingDirectory(self, "Ordner auswählen", start)
        else:
            value, _ = QFileDialog.getOpenFileName(self, "Datei auswählen", start, self.filter_text)
        if value:
            self.edit.setText(value)

    def path(self) -> Path | None:
        value = self.edit.text().strip()
        return Path(value) if value else None

    def set_path(self, path: Path | str | None) -> None:
        self.edit.setText(str(path) if path else "")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Audio-Werkbank · Markdown zu MP3")
        self.resize(1680, 980)
        self.setMinimumSize(1180, 720)

        self.settings = QSettings("Qwen3-TTS", "AudiobookBatch")
        self.source_path: Path | None = None
        self.source_text = ""
        self.segments: list[Segment] = []
        self.manual_splits: set[int] = set()
        self.included_by_key: dict[str, bool] = {}
        self.region_overrides_by_level: dict[int, dict[str, tuple[int, int]]] = {}
        self.preview_is_current = False
        self.processed_positions: dict[str, tuple[int, int]] = {}

        self.process: QProcess | None = None
        self.batch_queue: list[int] = []
        self.current_row: int | None = None
        self.cancel_requested = False
        self.completed_jobs = 0
        self.total_jobs = 0
        self.cooldown_started = 0.0
        self.last_gpu_check = 0.0
        self.gpu_unavailable_reported = False
        self.theme_name = "dark"

        self._build_ui()
        self._connect_signals()
        self._restore_settings()
        self._set_empty_state()

        self.cooldown_timer = QTimer(self)
        self.cooldown_timer.setInterval(500)
        self.cooldown_timer.timeout.connect(self._cooldown_tick)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        header = QFrame(objectName="header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 16, 24, 16)
        brand = QVBoxLayout()
        eyebrow = QLabel("QWEN3 · AUDIO PRODUCTION", objectName="eyebrow")
        title = QLabel("Audio-Werkbank", objectName="title")
        self.subtitle = QLabel("Markdown strukturieren, vorhörbar aufbereiten und kontrolliert rendern.", objectName="subtitle")
        brand.addWidget(eyebrow)
        brand.addWidget(title)
        brand.addWidget(self.subtitle)
        header_layout.addLayout(brand, 1)
        self.open_button = QPushButton("Markdown öffnen")
        self.open_button.setObjectName("accent")
        self.output_header_button = QPushButton("Ausgabeordner")
        self.theme_button = QPushButton("Heller Modus")
        self.theme_button.setToolTip("Zwischen heller und dunkler Darstellung wechseln")
        header_layout.addWidget(self.theme_button)
        header_layout.addWidget(self.output_header_button)
        header_layout.addWidget(self.open_button)
        root_layout.addWidget(header)

        workspace = QSplitter(Qt.Orientation.Horizontal)
        workspace.setChildrenCollapsible(False)
        workspace.addWidget(self._build_outline_panel())
        workspace.addWidget(self._build_preview_panel())
        workspace.addWidget(self._build_plan_panel())
        workspace.setSizes([310, 710, 570])
        root_layout.addWidget(workspace, 1)

        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.setChildrenCollapsible(False)
        bottom.addWidget(self._build_settings_tabs())
        bottom.addWidget(self._build_log_panel())
        bottom.setSizes([980, 610])
        bottom.setMaximumHeight(300)
        root_layout.addWidget(bottom)

        footer = QFrame()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(18, 10, 18, 12)
        self.batch_status = QLabel("Bereit", objectName="temperature")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("Noch kein Batch geplant")
        self.stop_button = QPushButton("Batch abbrechen", objectName="danger")
        self.start_button = QPushButton("Batch starten", objectName="accent")
        footer_layout.addWidget(self.batch_status)
        footer_layout.addWidget(self.progress, 1)
        footer_layout.addWidget(self.stop_button)
        footer_layout.addWidget(self.start_button)
        root_layout.addWidget(footer)

    def _panel(self, title: str, subtitle: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(9)
        layout.addWidget(QLabel(title, objectName="sectionTitle"))
        info = QLabel(subtitle, objectName="muted")
        info.setWordWrap(True)
        layout.addWidget(info)
        return frame, layout

    def _build_outline_panel(self) -> QWidget:
        frame, layout = self._panel("DOKUMENT", "Überschriften lassen sich einklappen. Ein Klick springt zur Textstelle.")
        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("Eine Datei pro"))
        self.level_combo = QComboBox()
        for level in range(1, 7):
            self.level_combo.addItem(f"H{level}", level)
        self.level_combo.setCurrentIndex(2)
        split_row.addWidget(self.level_combo, 1)
        layout.addLayout(split_row)
        self.heading_tree = QTreeWidget()
        self.heading_tree.setHeaderHidden(True)
        self.heading_tree.setAlternatingRowColors(True)
        layout.addWidget(self.heading_tree, 1)
        split_buttons = QHBoxLayout()
        self.add_split_button = QPushButton("Schnitt vor Absatz")
        self.undo_split_button = QPushButton("Letzten entfernen")
        split_buttons.addWidget(self.add_split_button)
        split_buttons.addWidget(self.undo_split_button)
        layout.addLayout(split_buttons)
        self.split_info = QLabel("Keine manuellen Schnitte", objectName="muted")
        layout.addWidget(self.split_info)
        return frame

    def _build_preview_panel(self) -> QWidget:
        frame, layout = self._panel("VORSCHAU", "Original und tatsächlich gesprochene Fassung bleiben jederzeit vergleichbar.")
        toolbar = QHBoxLayout()
        self.prepare_button = QPushButton("Vorlesetext aufbereiten", objectName="accent")
        self.reset_preview_button = QPushButton("Zurück zum Original")
        self.preview_badge = QLabel("NOCH NICHT AUFBEREITET", objectName="eyebrow")
        toolbar.addWidget(self.prepare_button)
        toolbar.addWidget(self.reset_preview_button)
        toolbar.addStretch(1)
        toolbar.addWidget(self.preview_badge)
        layout.addLayout(toolbar)
        self.preview_tabs = QTabWidget()
        self.original_preview = QPlainTextEdit()
        self.original_preview.setReadOnly(True)
        self.original_preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.speech_preview = QPlainTextEdit()
        self.speech_preview.setReadOnly(True)
        self.speech_preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        mono = QFont("Cascadia Mono", 10)
        self.original_preview.setFont(mono)
        self.speech_preview.setFont(mono)
        self.preview_tabs.addTab(self.original_preview, "Original")
        self.preview_tabs.addTab(self.speech_preview, "Vorlesetext")
        layout.addWidget(self.preview_tabs, 1)
        return frame

    def _build_plan_panel(self) -> QWidget:
        frame, layout = self._panel("DATEIPLAN", "Vor dem Start ist exakt sichtbar, welcher Text in welche MP3 gelangt.")
        controls = QHBoxLayout()
        self.select_all_button = QPushButton("Alle")
        self.select_none_button = QPushButton("Keine")
        self.plan_count = QLabel("0 Dateien", objectName="temperature")
        controls.addWidget(self.select_all_button)
        controls.addWidget(self.select_none_button)
        controls.addStretch(1)
        controls.addWidget(self.plan_count)
        layout.addLayout(controls)
        region_controls = QHBoxLayout()
        self.use_selection_button = QPushButton("Markierung übernehmen", objectName="accent")
        self.use_selection_button.setToolTip(
            "Den im Original markierten Text als Inhalt der ausgewählten MP3 speichern"
        )
        self.reset_region_button = QPushButton("Vorschlag wiederherstellen")
        self.reset_region_button.setToolTip(
            "Für die ausgewählte MP3 wieder den automatisch vorgeschlagenen Bereich verwenden"
        )
        self.region_state = QLabel("Bereich: Vorschlag", objectName="muted")
        self.use_selection_button.setEnabled(False)
        self.reset_region_button.setEnabled(False)
        region_controls.addWidget(self.use_selection_button)
        region_controls.addWidget(self.reset_region_button)
        region_controls.addStretch(1)
        region_controls.addWidget(self.region_state)
        layout.addLayout(region_controls)
        self.plan_table = QTableWidget(0, 5)
        self.plan_table.setHorizontalHeaderLabels(["✓", "Nr.", "Dateiname", "Zeichen", "Status"])
        self.plan_table.setAlternatingRowColors(True)
        self.plan_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.plan_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.plan_table.verticalHeader().setVisible(False)
        header = self.plan_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.plan_table, 1)
        return frame

    def _build_settings_tabs(self) -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._build_voice_settings(), "Stimme & Chunking")
        tabs.addTab(self._build_text_settings(), "Vorleseregeln")
        tabs.addTab(self._build_cooling_settings(), "Pause & GPU")
        return tabs

    def _build_voice_settings(self) -> QWidget:
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setColumnStretch(1, 1)
        self.voice_picker = PathPicker("file", "PyTorch-Embedding (*.pt);;Alle Dateien (*)")
        self.language_combo = QComboBox()
        self.language_combo.setEditable(True)
        self.language_combo.addItems(["German", "English", "French", "Spanish", "Italian", "Auto"])
        self.min_chunk = QSpinBox()
        self.min_chunk.setRange(50, 4000)
        self.min_chunk.setValue(220)
        self.max_chunk = QSpinBox()
        self.max_chunk.setRange(50, 4000)
        self.max_chunk.setValue(520)
        self.target_chunk_label = QLabel("Ziel: 370 Zeichen", objectName="temperature")
        self.output_picker = PathPicker("directory")
        grid.addWidget(QLabel("Stimme (.pt)"), 0, 0)
        grid.addWidget(self.voice_picker, 0, 1, 1, 5)
        grid.addWidget(QLabel("Sprache"), 1, 0)
        grid.addWidget(self.language_combo, 1, 1)
        grid.addWidget(QLabel("Chunk min."), 1, 2)
        grid.addWidget(self.min_chunk, 1, 3)
        grid.addWidget(QLabel("Chunk max."), 1, 4)
        grid.addWidget(self.max_chunk, 1, 5)
        grid.addWidget(QLabel("Ausgabe"), 2, 0)
        grid.addWidget(self.output_picker, 2, 1, 1, 4)
        grid.addWidget(self.target_chunk_label, 2, 5)
        return widget

    def _build_text_settings(self) -> QWidget:
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setColumnStretch(1, 1)
        self.dictionary_picker = PathPicker("file", "Aussprache-CSV (*.csv);;Alle Dateien (*)")
        self.replacements_picker = PathPicker("file", "Regel-CSV (*.csv);;Alle Dateien (*)")
        self.footnotes_checkbox = QCheckBox("Fußnoten direkt nach dem ersten Absatzverweis vorlesen")
        self.footnotes_checkbox.setChecked(True)
        self.hyphenate_checkbox = QCheckBox("Zusammengesetzte Zahlwörter für Qwen mit Bindestrichen gliedern")
        self.hyphenate_checkbox.setChecked(True)
        example = QLabel(f"Beispielregeln: {APP_DIR / 'project_replacements.example.csv'}", objectName="muted")
        example.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        grid.addWidget(QLabel("Aussprachewörterbuch"), 0, 0)
        grid.addWidget(self.dictionary_picker, 0, 1)
        grid.addWidget(QLabel("Projektregeln"), 1, 0)
        grid.addWidget(self.replacements_picker, 1, 1)
        grid.addWidget(self.footnotes_checkbox, 2, 0, 1, 2)
        grid.addWidget(self.hyphenate_checkbox, 3, 0, 1, 2)
        grid.addWidget(example, 4, 0, 1, 2)
        return widget

    def _build_cooling_settings(self) -> QWidget:
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(14, 12, 14, 12)
        self.pause_seconds = QDoubleSpinBox()
        self.pause_seconds.setRange(0, 3600)
        self.pause_seconds.setDecimals(1)
        self.pause_seconds.setSuffix(" s")
        self.pause_seconds.setValue(10.0)
        self.temperature_checkbox = QCheckBox("Zusätzlich auf NVIDIA-Zieltemperatur warten")
        self.temperature_checkbox.setChecked(True)
        self.temperature_limit = QSpinBox()
        self.temperature_limit.setRange(30, 95)
        self.temperature_limit.setSuffix(" °C")
        self.temperature_limit.setValue(65)
        self.gpu_index = QSpinBox()
        self.gpu_index.setRange(0, 15)
        self.gpu_index.setValue(0)
        note = QLabel(
            "Der nächste Auftrag startet erst nach der Mindestpause und sobald die GPU unter dem Grenzwert liegt. "
            "Es gibt bewusst keine maximale Wartezeit.",
            objectName="muted",
        )
        note.setWordWrap(True)
        grid.addWidget(QLabel("Mindestpause"), 0, 0)
        grid.addWidget(self.pause_seconds, 0, 1)
        grid.addWidget(self.temperature_checkbox, 1, 0, 1, 2)
        grid.addWidget(QLabel("Grenzwert: unter"), 2, 0)
        grid.addWidget(self.temperature_limit, 2, 1)
        grid.addWidget(QLabel("NVIDIA-GPU"), 2, 2)
        grid.addWidget(self.gpu_index, 2, 3)
        grid.addWidget(note, 3, 0, 1, 4)
        grid.setColumnStretch(4, 1)
        return widget

    def _build_log_panel(self) -> QWidget:
        frame, layout = self._panel("BATCH-PROTOKOLL", "Ausgabe des Generators und Status der Abkühlphase.")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setFont(QFont("Cascadia Mono", 9))
        layout.addWidget(self.log, 1)
        return frame

    def _connect_signals(self) -> None:
        self.open_button.clicked.connect(self.open_markdown_dialog)
        self.theme_button.clicked.connect(self._toggle_theme)
        self.output_header_button.clicked.connect(self.output_picker._browse)
        self.level_combo.currentIndexChanged.connect(self._rebuild_segments)
        self.heading_tree.itemSelectionChanged.connect(self._heading_selected)
        self.add_split_button.clicked.connect(self._add_manual_split)
        self.undo_split_button.clicked.connect(self._undo_manual_split)
        self.prepare_button.clicked.connect(self.prepare_preview)
        self.reset_preview_button.clicked.connect(self.reset_preview)
        self.select_all_button.clicked.connect(lambda: self._set_all_included(True))
        self.select_none_button.clicked.connect(lambda: self._set_all_included(False))
        self.plan_table.itemChanged.connect(self._plan_item_changed)
        self.plan_table.itemSelectionChanged.connect(self._plan_selection_changed)
        self.use_selection_button.clicked.connect(self._use_original_selection)
        self.reset_region_button.clicked.connect(self._reset_selected_region)
        self.min_chunk.valueChanged.connect(self._update_target_chunk)
        self.max_chunk.valueChanged.connect(self._update_target_chunk)
        self.dictionary_picker.changed.connect(self._mark_preview_stale)
        self.replacements_picker.changed.connect(self._mark_preview_stale)
        self.footnotes_checkbox.toggled.connect(self._mark_preview_stale)
        self.hyphenate_checkbox.toggled.connect(self._mark_preview_stale)
        self.start_button.clicked.connect(self.start_batch)
        self.stop_button.clicked.connect(self.stop_batch)

    def _restore_settings(self) -> None:
        self._apply_theme(str(self.settings.value("theme", "dark")))
        self.voice_picker.set_path(self.settings.value("voice", self._default_voice_path()))
        self.language_combo.setCurrentText(str(self.settings.value("language", "German")))
        self.min_chunk.setValue(int(self.settings.value("min_chunk", 220)))
        self.max_chunk.setValue(int(self.settings.value("max_chunk", 520)))
        self.dictionary_picker.set_path(self.settings.value("dictionary", ""))
        self.replacements_picker.set_path(self.settings.value("replacements", ""))
        self.pause_seconds.setValue(float(self.settings.value("pause_seconds", 10.0)))
        self.temperature_limit.setValue(int(self.settings.value("temperature_limit", 65)))
        self.gpu_index.setValue(int(self.settings.value("gpu_index", 0)))
        self._update_target_chunk()

    def _save_settings(self) -> None:
        self.settings.setValue("theme", self.theme_name)
        self.settings.setValue("voice", self.voice_picker.edit.text())
        self.settings.setValue("language", self.language_combo.currentText())
        self.settings.setValue("min_chunk", self.min_chunk.value())
        self.settings.setValue("max_chunk", self.max_chunk.value())
        self.settings.setValue("dictionary", self.dictionary_picker.edit.text())
        self.settings.setValue("replacements", self.replacements_picker.edit.text())
        self.settings.setValue("pause_seconds", self.pause_seconds.value())
        self.settings.setValue("temperature_limit", self.temperature_limit.value())
        self.settings.setValue("gpu_index", self.gpu_index.value())
        if self.source_path:
            self.settings.setValue("last_source", str(self.source_path))

    def _apply_theme(self, theme: str) -> None:
        self.theme_name = theme if theme in {"light", "dark"} else "dark"
        application = QApplication.instance()
        if application:
            application.setStyleSheet(LIGHT_STYLE if self.theme_name == "light" else DARK_STYLE)
        self.theme_button.setText("Dunkler Modus" if self.theme_name == "light" else "Heller Modus")

    def _toggle_theme(self) -> None:
        self._apply_theme("light" if self.theme_name == "dark" else "dark")
        self._save_settings()

    def _default_voice_path(self) -> str:
        preferred = DEFAULT_VOICE_DIR / "Sachbuch-Autor-1.7B.pt"
        if preferred.is_file():
            return str(preferred)
        choices = sorted(DEFAULT_VOICE_DIR.glob("*.pt")) if DEFAULT_VOICE_DIR.is_dir() else []
        return str(choices[0]) if choices else ""

    def _set_empty_state(self) -> None:
        enabled = bool(self.source_text)
        for widget in (
            self.heading_tree,
            self.original_preview,
            self.speech_preview,
            self.plan_table,
            self.prepare_button,
            self.reset_preview_button,
            self.add_split_button,
            self.undo_split_button,
            self.start_button,
        ):
            widget.setEnabled(enabled)
        self.stop_button.setEnabled(False)

    def open_markdown_dialog(self) -> None:
        start = str(self.source_path.parent if self.source_path else DEFAULT_INPUT_DIR)
        value, _ = QFileDialog.getOpenFileName(self, "Markdown öffnen", start, "Markdown (*.md);;Text (*.txt)")
        if value:
            self.load_markdown(Path(value))

    def load_markdown(self, path: Path) -> None:
        try:
            text = read_utf8_text(path)
        except (OSError, UnicodeError) as error:
            QMessageBox.critical(self, "Datei konnte nicht geöffnet werden", str(error))
            return
        self.source_path = path.resolve()
        self.source_text = text
        self.manual_splits.clear()
        self.included_by_key.clear()
        self._load_region_overrides()
        self.original_preview.setPlainText(text)
        self.speech_preview.clear()
        self.subtitle.setText(f"{self.source_path.name} · {len(text):,} Zeichen".replace(",", "."))
        if not self.output_picker.edit.text().strip():
            self.output_picker.set_path(path.parent / f"audio_{path.stem}")
        self._populate_heading_tree()
        self._rebuild_segments()
        self._set_empty_state()
        self._save_settings()
        self._append_log(f"Geöffnet: {self.source_path}")

    def _populate_heading_tree(self) -> None:
        self.heading_tree.clear()
        stack: list[tuple[int, QTreeWidgetItem]] = []
        for heading in parse_headings(self.source_text):
            item = QTreeWidgetItem([f"H{heading.level}  {heading.title}"])
            item.setData(0, Qt.ItemDataRole.UserRole, heading.start)
            item.setToolTip(0, f"Zeile {heading.line}")
            while stack and stack[-1][0] >= heading.level:
                stack.pop()
            if stack:
                stack[-1][1].addChild(item)
            else:
                self.heading_tree.addTopLevelItem(item)
            stack.append((heading.level, item))
        self.heading_tree.expandToDepth(1)

    def _heading_selected(self) -> None:
        items = self.heading_tree.selectedItems()
        if not items:
            return
        position = int(items[0].data(0, Qt.ItemDataRole.UserRole))
        cursor = self.original_preview.textCursor()
        cursor.setPosition(position)
        self.original_preview.setTextCursor(cursor)
        self.original_preview.centerCursor()
        for row, segment in enumerate(self.segments):
            if segment.start <= position < segment.end:
                self.plan_table.selectRow(row)
                break

    def _rebuild_segments(self) -> None:
        if not self.source_text:
            return
        for row, segment in enumerate(self.segments):
            item = self.plan_table.item(row, 0)
            if item:
                self.included_by_key[segment.key] = item.checkState() == Qt.CheckState.Checked
        level = int(self.level_combo.currentData())
        self.segments = build_segments(self.source_text, level, self.manual_splits)
        overrides = self.region_overrides_by_level.get(level, {})
        invalid_keys: list[str] = []
        for segment in self.segments:
            custom_range = overrides.get(segment.key)
            if not custom_range:
                continue
            try:
                set_segment_range(segment, self.source_text, *custom_range)
                segment.status = "Bereich angepasst"
            except ValueError:
                invalid_keys.append(segment.key)
        for key in invalid_keys:
            overrides.pop(key, None)
        if invalid_keys:
            self._save_region_overrides()
        included = [self.included_by_key.get(item.key, item.included_by_default) for item in self.segments]
        assign_output_names(self.segments, included)
        self._populate_plan(included)
        self._mark_preview_stale()
        self.split_info.setText(
            f"{len(self.manual_splits)} manuelle Schnittmarke(n)" if self.manual_splits else "Keine manuellen Schnitte"
        )

    def _populate_plan(self, included: list[bool]) -> None:
        self.plan_table.blockSignals(True)
        self.plan_table.setRowCount(len(self.segments))
        for row, (segment, selected) in enumerate(zip(self.segments, included)):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
            check.setCheckState(Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked)
            check.setData(Qt.ItemDataRole.UserRole, segment.key)
            number = QTableWidgetItem(str(row + 1))
            filename = QTableWidgetItem(segment.output_name)
            chars = QTableWidgetItem(str(len(segment.source_text)))
            status = QTableWidgetItem(segment.status)
            for item in (number, filename, chars, status):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.plan_table.setItem(row, 0, check)
            self.plan_table.setItem(row, 1, number)
            self.plan_table.setItem(row, 2, filename)
            self.plan_table.setItem(row, 3, chars)
            self.plan_table.setItem(row, 4, status)
        self.plan_table.blockSignals(False)
        self._update_plan_names()

    def _included_states(self) -> list[bool]:
        return [
            bool(self.plan_table.item(row, 0) and self.plan_table.item(row, 0).checkState() == Qt.CheckState.Checked)
            for row in range(len(self.segments))
        ]

    def _update_plan_names(self) -> None:
        included = self._included_states()
        assign_output_names(self.segments, included)
        self.plan_table.blockSignals(True)
        for row, segment in enumerate(self.segments):
            self.plan_table.item(row, 1).setText(str(sum(included[: row + 1])) if included[row] else "—")
            self.plan_table.item(row, 2).setText(segment.output_name)
        self.plan_table.blockSignals(False)
        count = sum(included)
        self.plan_count.setText(f"{count} Datei{'en' if count != 1 else ''}")
        self.progress.setRange(0, max(1, count))
        self.progress.setFormat(f"{count} Dateien geplant")

    def _plan_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        key = str(item.data(Qt.ItemDataRole.UserRole))
        self.included_by_key[key] = item.checkState() == Qt.CheckState.Checked
        self._update_plan_names()

    def _set_all_included(self, value: bool) -> None:
        for row in range(self.plan_table.rowCount()):
            self.plan_table.item(row, 0).setCheckState(
                Qt.CheckState.Checked if value else Qt.CheckState.Unchecked
            )

    def _plan_selection_changed(self) -> None:
        row = self.plan_table.currentRow()
        if not 0 <= row < len(self.segments):
            self.use_selection_button.setEnabled(False)
            self.reset_region_button.setEnabled(False)
            self.region_state.setText("Bereich: Vorschlag")
            return
        segment = self.segments[row]
        overrides = self.region_overrides_by_level.get(int(self.level_combo.currentData()), {})
        is_custom = segment.key in overrides
        self.use_selection_button.setEnabled(True)
        self.reset_region_button.setEnabled(is_custom)
        self.region_state.setText("Bereich: individuell gespeichert" if is_custom else "Bereich: Vorschlag")
        cursor = self.original_preview.textCursor()
        # Build the selection backwards.  It covers the same text, but leaves
        # the active cursor at the beginning so centerCursor() scrolls there.
        cursor.setPosition(segment.end)
        cursor.setPosition(segment.start, QTextCursor.MoveMode.KeepAnchor)
        self.original_preview.setTextCursor(cursor)
        self.original_preview.centerCursor()
        if segment.key in self.processed_positions:
            start, end = self.processed_positions[segment.key]
            speech_cursor = self.speech_preview.textCursor()
            speech_cursor.setPosition(end)
            speech_cursor.setPosition(start, QTextCursor.MoveMode.KeepAnchor)
            self.speech_preview.setTextCursor(speech_cursor)
            self.speech_preview.centerCursor()

    def _region_settings_key(self) -> str | None:
        if not self.source_path:
            return None
        source_id = hashlib.sha256(str(self.source_path).casefold().encode("utf-8")).hexdigest()[:24]
        return f"region_overrides/{source_id}"

    def _load_region_overrides(self) -> None:
        self.region_overrides_by_level = {}
        key = self._region_settings_key()
        if not key or not self.source_text:
            return
        raw = self.settings.value(key, "")
        if not raw:
            return
        try:
            payload = json.loads(str(raw))
            if not isinstance(payload, dict):
                return
            fingerprint = hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()
            if payload.get("source") != str(self.source_path) or payload.get("fingerprint") != fingerprint:
                return
            raw_levels = payload.get("levels", {})
            if not isinstance(raw_levels, dict):
                return
            for raw_level, raw_ranges in raw_levels.items():
                level = int(raw_level)
                if not 1 <= level <= 6 or not isinstance(raw_ranges, dict):
                    continue
                parsed: dict[str, tuple[int, int]] = {}
                for segment_key, bounds in raw_ranges.items():
                    if (
                        isinstance(segment_key, str)
                        and isinstance(bounds, list)
                        and len(bounds) == 2
                        and all(isinstance(value, int) for value in bounds)
                    ):
                        parsed[segment_key] = (bounds[0], bounds[1])
                if parsed:
                    self.region_overrides_by_level[level] = parsed
        except (TypeError, ValueError, json.JSONDecodeError):
            self.region_overrides_by_level = {}

    def _save_region_overrides(self) -> None:
        key = self._region_settings_key()
        if not key or not self.source_path:
            return
        levels = {
            str(level): {segment_key: [start, end] for segment_key, (start, end) in ranges.items()}
            for level, ranges in self.region_overrides_by_level.items()
            if ranges
        }
        if not levels:
            self.settings.remove(key)
            return
        payload = {
            "source": str(self.source_path),
            "fingerprint": hashlib.sha256(self.source_text.encode("utf-8")).hexdigest(),
            "levels": levels,
        }
        self.settings.setValue(key, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    def _select_plan_row_by_key(self, segment_key: str) -> None:
        for row, segment in enumerate(self.segments):
            if segment.key == segment_key:
                self.plan_table.setCurrentCell(row, 2)
                self.plan_table.selectRow(row)
                return

    def _use_original_selection(self) -> None:
        row = self.plan_table.currentRow()
        if not 0 <= row < len(self.segments):
            return
        cursor = self.original_preview.textCursor()
        start, end = cursor.selectionStart(), cursor.selectionEnd()
        try:
            set_segment_range(self.segments[row], self.source_text, start, end)
        except ValueError as error:
            QMessageBox.information(self, "Bereich nicht übernommen", str(error))
            return
        segment_key = self.segments[row].key
        title = self.segments[row].title
        level = int(self.level_combo.currentData())
        self.region_overrides_by_level.setdefault(level, {})[segment_key] = (start, end)
        self._save_region_overrides()
        self._rebuild_segments()
        self._select_plan_row_by_key(segment_key)
        self._append_log(f"Individueller Bereich gespeichert: {title} · {end - start} Zeichen")

    def _reset_selected_region(self) -> None:
        row = self.plan_table.currentRow()
        if not 0 <= row < len(self.segments):
            return
        segment_key = self.segments[row].key
        level = int(self.level_combo.currentData())
        overrides = self.region_overrides_by_level.get(level, {})
        if segment_key not in overrides:
            return
        overrides.pop(segment_key, None)
        if not overrides:
            self.region_overrides_by_level.pop(level, None)
        self._save_region_overrides()
        self._rebuild_segments()
        self._select_plan_row_by_key(segment_key)
        self._append_log("Automatisch vorgeschlagener Bereich wiederhergestellt.")

    def _add_manual_split(self) -> None:
        if not self.source_text:
            return
        position = paragraph_start(self.source_text, self.original_preview.textCursor().position())
        if position <= 0 or position >= len(self.source_text):
            QMessageBox.information(self, "Keine Schnittmarke", "Bitte den Cursor in einen späteren Absatz setzen.")
            return
        self.manual_splits.add(position)
        self._rebuild_segments()
        self._highlight_manual_splits()

    def _undo_manual_split(self) -> None:
        if not self.manual_splits:
            return
        self.manual_splits.remove(max(self.manual_splits))
        self._rebuild_segments()
        self._highlight_manual_splits()

    def _highlight_manual_splits(self) -> None:
        selections = []
        for position in sorted(self.manual_splits):
            cursor = self.original_preview.textCursor()
            cursor.setPosition(position)
            cursor.select(QTextCursor.SelectionType.LineUnderCursor)
            selection = QPlainTextEdit.ExtraSelection()
            selection.cursor = cursor
            selection.format = QTextCharFormat()
            selection.format.setBackground(QColor("#513d25"))
            selection.format.setProperty(QTextCharFormat.Property.FullWidthSelection, True)
            selections.append(selection)
        self.original_preview.setExtraSelections(selections)

    def _speech_options(self) -> SpeechOptions:
        dictionary = self.dictionary_picker.path()
        replacements = self.replacements_picker.path()
        pronunciation_entries = load_pronunciation_csv(dictionary) if dictionary else []
        project_entries = load_project_replacements(replacements) if replacements else []
        return SpeechOptions(
            speak_footnotes=self.footnotes_checkbox.isChecked(),
            pronunciation_entries=pronunciation_entries,
            project_replacements=project_entries,
            hyphenate_number_words=self.hyphenate_checkbox.isChecked(),
        )

    def prepare_preview(self) -> bool:
        if not self.source_text:
            return False
        try:
            options = self._speech_options()
            prepared = prepare_segments_for_speech(self.source_text, self.segments, options)
        except (OSError, UnicodeError, ValueError) as error:
            QMessageBox.critical(self, "Vorleseregeln konnten nicht angewendet werden", str(error))
            return False
        pieces: list[str] = []
        self.processed_positions.clear()
        cursor = 0
        separator = "\n\n──────────\n\n"
        for segment, text in zip(self.segments, prepared):
            if pieces:
                pieces.append(separator)
                cursor += len(separator)
            start = cursor
            pieces.append(text)
            cursor += len(text)
            self.processed_positions[segment.key] = (start, cursor)
        self.speech_preview.setPlainText("".join(pieces))
        self.preview_is_current = True
        self.preview_badge.setText("VORLESETEXT AKTUELL")
        self.preview_badge.setStyleSheet("color: #73d6cb;")
        self.preview_tabs.setCurrentWidget(self.speech_preview)
        for row, segment in enumerate(self.segments):
            self.plan_table.item(row, 3).setText(str(len(segment.speech_text)))
        self._append_log(
            f"Vorlesetext aufbereitet · {len(options.pronunciation_entries)} Ausspracheeinträge · "
            f"{len(options.project_replacements)} Projektregeln"
        )
        return True

    def reset_preview(self) -> None:
        self.speech_preview.clear()
        for segment in self.segments:
            segment.speech_text = ""
        self.processed_positions.clear()
        self.preview_is_current = False
        self.preview_badge.setText("ORIGINAL AKTIV")
        self.preview_badge.setStyleSheet("")
        self.preview_tabs.setCurrentWidget(self.original_preview)
        for row, segment in enumerate(self.segments):
            self.plan_table.item(row, 3).setText(str(len(segment.source_text)))

    def _mark_preview_stale(self, *_args: object) -> None:
        if self.preview_is_current:
            self.preview_is_current = False
            self.preview_badge.setText("REGELN GEÄNDERT · NEU AUFBEREITEN")
            self.preview_badge.setStyleSheet("color: #e88a6b;")

    def _update_target_chunk(self) -> None:
        target = (self.min_chunk.value() + self.max_chunk.value()) // 2
        self.target_chunk_label.setText(f"Ziel: {target} Zeichen")

    def _validate_batch(self) -> tuple[Path, Path, list[int]] | None:
        if not self.source_path:
            QMessageBox.warning(self, "Keine Quelle", "Bitte zuerst eine Markdown-Datei öffnen.")
            return None
        output = self.output_picker.path()
        voice = self.voice_picker.path()
        if not output:
            QMessageBox.warning(self, "Kein Ausgabeordner", "Bitte einen Ausgabeordner auswählen.")
            return None
        if not voice or not voice.is_file():
            QMessageBox.warning(self, "Stimme fehlt", "Bitte ein vorhandenes .pt-Sprecher-Embedding auswählen.")
            return None
        if self.min_chunk.value() > self.max_chunk.value():
            QMessageBox.warning(self, "Chunk-Größen ungültig", "Minimum darf Maximum nicht überschreiten.")
            return None
        rows = [row for row, value in enumerate(self._included_states()) if value]
        if not rows:
            QMessageBox.warning(self, "Leerer Batch", "Bitte mindestens einen Abschnitt auswählen.")
            return None
        if not GENERATOR_SCRIPT.is_file() or not BASE_CONFIG.is_file():
            QMessageBox.critical(self, "Generator fehlt", "Generator oder config.json wurde nicht gefunden.")
            return None
        return output.resolve(), voice.resolve(), rows

    def start_batch(self) -> None:
        if self.process or self.cooldown_timer.isActive():
            return
        validated = self._validate_batch()
        if not validated:
            return
        output, voice, rows = validated
        if not self.preview_is_current and not self.prepare_preview():
            return
        output.mkdir(parents=True, exist_ok=True)
        session = output / ".audiobook_batch"
        session.mkdir(parents=True, exist_ok=True)
        try:
            self.session_config = create_batch_config(
                BASE_CONFIG,
                session / "config.json",
                speaker=voice,
                language=self.language_combo.currentText(),
                min_chunk_chars=self.min_chunk.value(),
                max_chunk_chars=self.max_chunk.value(),
            )
            for row in rows:
                (session / f"segment_{row + 1:04d}.md").write_text(
                    self.segments[row].speech_text + "\n", encoding="utf-8"
                )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            QMessageBox.critical(self, "Batch konnte nicht vorbereitet werden", str(error))
            return

        self.batch_output = output
        self.batch_session = session
        self.batch_queue = []
        self.completed_jobs = 0
        for row in rows:
            target = output / self.segments[row].output_name
            if target.exists():
                self._set_row_status(row, "Übersprungen")
                self.completed_jobs += 1
            else:
                self.batch_queue.append(row)
                self._set_row_status(row, "Wartet")
        self.total_jobs = len(rows)
        self.cancel_requested = False
        self.progress.setRange(0, max(1, self.total_jobs))
        self.progress.setValue(self.completed_jobs)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._save_settings()
        self._write_manifest()
        self._append_log(f"Batch gestartet: {self.total_jobs} geplante Dateien · Ausgabe {output}")
        self._start_next_job()

    def _start_next_job(self) -> None:
        if self.cancel_requested:
            self._finish_batch("Abgebrochen")
            return
        if not self.batch_queue:
            self._finish_batch("Batch abgeschlossen")
            return
        row = self.batch_queue.pop(0)
        self.current_row = row
        segment = self.segments[row]
        source = self.batch_session / f"segment_{row + 1:04d}.md"
        output = self.batch_output / segment.output_name
        self._set_row_status(row, "Wird erzeugt")
        self.batch_status.setText(f"Erzeuge {segment.output_name}")
        self._append_log(f"\n▶ {segment.output_name}")

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(
            [
                "-u",
                str(GENERATOR_SCRIPT),
                "--config",
                str(self.session_config),
                "--input",
                str(source),
                "--output",
                str(output),
            ]
        )
        process.setWorkingDirectory(str(REPO_ROOT))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_process_output)
        process.finished.connect(self._process_finished)
        process.errorOccurred.connect(self._process_error)
        self.process = process
        process.start()

    def _read_process_output(self) -> None:
        if not self.process:
            return
        value = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if value:
            self.log.moveCursor(QTextCursor.MoveOperation.End)
            self.log.insertPlainText(value)
            self.log.moveCursor(QTextCursor.MoveOperation.End)

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._append_log("Generator konnte nicht gestartet werden.")

    def _process_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        row = self.current_row
        self.process = None
        self.current_row = None
        if row is None:
            return
        if self.cancel_requested:
            self._set_row_status(row, "Abgebrochen")
            self._finish_batch("Abgebrochen")
            return
        if exit_code != 0:
            self._set_row_status(row, f"Fehler ({exit_code})")
            self._write_manifest()
            self._finish_batch("Batch wegen Fehler angehalten")
            QMessageBox.critical(
                self,
                "Erzeugung fehlgeschlagen",
                f"{self.segments[row].output_name} schlug fehl. Details stehen im Protokoll.",
            )
            return
        self.completed_jobs += 1
        self.progress.setValue(self.completed_jobs)
        self._set_row_status(row, "Fertig")
        self._write_manifest()
        if not self.batch_queue:
            self._finish_batch("Batch abgeschlossen")
            return
        self.cooldown_started = time.monotonic()
        self.last_gpu_check = 0.0
        self.gpu_unavailable_reported = False
        self.cooldown_timer.start()
        self._cooldown_tick()

    def _cooldown_tick(self) -> None:
        elapsed = time.monotonic() - self.cooldown_started
        remaining = max(0.0, self.pause_seconds.value() - elapsed)
        temperature: int | None = None
        if self.temperature_checkbox.isChecked() and time.monotonic() - self.last_gpu_check >= 2.0:
            self.last_gpu_check = time.monotonic()
            temperature = self._read_gpu_temperature()
            self._last_temperature = temperature
        else:
            temperature = getattr(self, "_last_temperature", None)

        pause_ready = remaining <= 0
        temperature_ready = not self.temperature_checkbox.isChecked()
        if self.temperature_checkbox.isChecked():
            if temperature is None:
                temperature_ready = True
                if not self.gpu_unavailable_reported:
                    self.gpu_unavailable_reported = True
                    self._append_log("NVIDIA-Temperatur nicht verfügbar; es gilt nur die Mindestpause.")
            else:
                temperature_ready = temperature < self.temperature_limit.value()

        temp_text = f" · GPU {temperature} °C" if temperature is not None else ""
        self.batch_status.setText(f"Abkühlung · noch {remaining:.1f} s{temp_text}")
        if pause_ready and temperature_ready:
            self.cooldown_timer.stop()
            self._append_log(f"Abkühlung beendet{temp_text}.")
            self._start_next_job()

    def _read_gpu_temperature(self) -> int | None:
        command = [
            "nvidia-smi",
            f"--id={self.gpu_index.value()}",
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        kwargs: dict[str, object] = {
            "capture_output": True,
            "text": True,
            "timeout": 3,
            "check": False,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            result = subprocess.run(command, **kwargs)
            if result.returncode != 0:
                return None
            return int(result.stdout.strip().splitlines()[0])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None

    def stop_batch(self) -> None:
        self.cancel_requested = True
        self.cooldown_timer.stop()
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self._append_log("Abbruch angefordert …")
            self.process.kill()
        else:
            self._finish_batch("Abgebrochen")

    def _finish_batch(self, message: str) -> None:
        self.cooldown_timer.stop()
        self.process = None
        self.batch_queue.clear()
        self.current_row = None
        self.start_button.setEnabled(bool(self.source_text))
        self.stop_button.setEnabled(False)
        self.batch_status.setText(message)
        self.progress.setFormat(f"{self.completed_jobs}/{self.total_jobs} · {message}")
        self._append_log(message)
        self._write_manifest()

    def _set_row_status(self, row: int, status: str) -> None:
        self.segments[row].status = status
        item = self.plan_table.item(row, 4)
        if item:
            item.setText(status)
            colors = {
                "Fertig": "#73d6cb",
                "Wird erzeugt": "#f2c879",
                "Übersprungen": "#8ea0a8",
                "Abgebrochen": "#e88a6b",
            }
            item.setForeground(QColor(colors.get(status, "#e8e4dc")))

    def _write_manifest(self) -> None:
        session = getattr(self, "batch_session", None)
        if not session:
            return
        data = {
            "source": str(self.source_path) if self.source_path else "",
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "split_level": self.level_combo.currentData(),
            "manual_splits": sorted(self.manual_splits),
            "files": [
                {
                    "title": segment.title,
                    "output": segment.output_name,
                    "status": segment.status,
                    "included": included,
                    "source_start": segment.start,
                    "source_end": segment.end,
                    "custom_range": segment.key
                    in self.region_overrides_by_level.get(int(self.level_combo.currentData()), {}),
                }
                for segment, included in zip(self.segments, self._included_states())
            ],
        }
        try:
            (session / "batch_plan.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as error:
            self._append_log(f"Batch-Plan konnte nicht gespeichert werden: {error}")

    def _append_log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{stamp}] {message}")

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            answer = QMessageBox.question(
                self,
                "Batch läuft",
                "Der aktuelle Generatorprozess wird beim Schließen beendet. Wirklich schließen?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.process.kill()
        self._save_settings()
        event.accept()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", type=Path, help="Markdown file to open at startup")
    parser.add_argument("--screenshot", type=Path, help="Save a UI screenshot and exit (for visual checks)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    application = QApplication(sys.argv[:1])
    application.setApplicationName("Audio-Werkbank")
    application.setOrganizationName("Qwen3-TTS")
    application.setStyle("Fusion")
    for font_path in (
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\seguisb.ttf"),
        Path(r"C:\Windows\Fonts\bahnschrift.ttf"),
        Path(r"C:\Windows\Fonts\CascadiaMono.ttf"),
    ):
        if font_path.is_file():
            QFontDatabase.addApplicationFont(str(font_path))
    application.setStyleSheet(DARK_STYLE)
    window = MainWindow()
    window.show()
    if args.open:
        QTimer.singleShot(0, lambda: window.load_markdown(args.open))
    if args.screenshot:
        def save_screenshot() -> None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.screenshot))
            application.quit()

        QTimer.singleShot(1500, save_screenshot)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
