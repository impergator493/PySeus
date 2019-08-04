"""Contains the GUI elements for PySeus."""

from .main import MainWindow
from .view import ViewWidget


def get_stylesheet():
    """Provides the custom stylesheet for PySeus."""
    # @TODO refactor into separate module / file --> include_package_data

    return """
QMenuBar { background: #111; color: #ddd; }
QMenuBar::item { padding: 5px 10px 5px 10px; }
QMenuBar::item:selected { background: #222; }
QMenu { background: #222; color: #eee; padding: 0px; }
QMenu::item { padding: 5px 10px 5px 10px; }
QMenu::item:selected { background: #333; }

QScrollArea {
    background: #111;
    border: none;
}

QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 2px 10px 2px 10px;
}
QScrollBar::handle:horizontal {
    background-color: #bbb;
    min-width: 12px;
    border-radius: 2px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #eee;
}

QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 10px 2px 10px 2px;
}
QScrollBar::handle:vertical {
    background-color: #bbb;
    min-height: 12px;
    border-radius: 2px;
}
QScrollBar::handle:vertical:hover {
    background-color: #eee;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}

QLabel { background: #111; }

QStatusBar { background: #111; color: #eee; }
"""
