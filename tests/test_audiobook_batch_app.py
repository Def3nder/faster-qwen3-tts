from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from generate.audiobook_batch.app import MainWindow


def test_batch_lock_preserves_navigation_and_blocks_mutations(monkeypatch) -> None:
    monkeypatch.setattr(MainWindow, "_read_gpu_temperature", lambda self: 55)
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.source_path = Path("test.md").resolve()
        window.source_text = "### Eins\n\nText\n\n### Zwei\n\nText"
        window.level_combo.setCurrentIndex(window.level_combo.findData(3))
        window._rebuild_segments()
        window._set_empty_state()
        window.plan_table.selectRow(0)
        window.stop_button.setEnabled(True)

        window._set_batch_edit_lock(True)

        assert window.plan_table.isEnabled()
        assert window.heading_tree.isEnabled()
        assert window.preview_tabs.isEnabled()
        assert window.log.isEnabled()
        assert window.theme_button.isEnabled()
        assert window.stop_button.isEnabled()
        window.plan_table.selectRow(1)
        assert window.plan_table.currentRow() == 1

        mutating_widgets = (
            window.open_button,
            window.output_header_button,
            window.level_combo,
            window.add_split_button,
            window.undo_split_button,
            window.prepare_button,
            window.reset_preview_button,
            window.select_all_button,
            window.select_none_button,
            window.use_selection_button,
            window.reset_region_button,
            window.voice_picker,
            window.language_combo,
            window.min_chunk,
            window.max_chunk,
            window.output_picker,
            window.dictionary_picker,
            window.replacements_picker,
            window.footnotes_checkbox,
            window.hyphenate_checkbox,
            window.pause_seconds,
            window.temperature_checkbox,
            window.temperature_limit,
            window.gpu_index,
        )
        assert all(not widget.isEnabled() for widget in mutating_widgets)
        assert not bool(
            window.plan_table.item(0, 0).flags() & Qt.ItemFlag.ItemIsUserCheckable
        )
        assert not bool(
            window.plan_table.item(0, 2).flags() & Qt.ItemFlag.ItemIsEditable
        )

        window._set_batch_edit_lock(False)

        assert window.open_button.isEnabled()
        assert window.prepare_button.isEnabled()
        assert window.voice_picker.isEnabled()
        assert window.pause_seconds.isEnabled()
        assert bool(
            window.plan_table.item(0, 0).flags() & Qt.ItemFlag.ItemIsUserCheckable
        )
        assert bool(
            window.plan_table.item(0, 2).flags() & Qt.ItemFlag.ItemIsEditable
        )
    finally:
        window.gpu_timer.stop()
        window.close()
        application.processEvents()


def test_running_progress_text_and_gpu_timer_survive_batch_finish(monkeypatch) -> None:
    monkeypatch.setattr(MainWindow, "_read_gpu_temperature", lambda self: 55)
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.total_jobs = 62
        window.completed_jobs = 21
        window._show_running_progress()

        assert window.progress.value() == 21
        assert window.progress.maximum() == 62
        assert window.progress.text() == "21/62"

        window.completed_jobs = 22
        window.progress.setValue(window.completed_jobs)
        assert window.progress.text() == "22/62"

        assert window.gpu_timer.isActive()
        window._finish_batch("Abgebrochen")
        assert window.gpu_timer.isActive()
    finally:
        window.gpu_timer.stop()
        window.close()
        application.processEvents()
