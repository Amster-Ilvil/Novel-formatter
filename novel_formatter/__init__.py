"""Headless entry points for Novel Formatter.

This package intentionally contains no Qt imports.  The desktop UI remains a
frontend over the same engine/model modules, while automation and Codex can use
``python -m novel_formatter.cli`` without creating QApplication/MainWindow.
"""

__all__ = []
