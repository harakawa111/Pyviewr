"""Application entrypoint."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from pyviewr.ui.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    args = sys.argv if argv is None else argv
    app = QApplication(args)
    app.setApplicationName("Pyviewr")
    window = MainWindow()
    window.show()
    return app.exec()
