from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QFrame

from ui.common.styling import MUTED, wrap_in_card
from ui.pages.tab import PageManagerTab
from ui.workspace.dashboard import ProjectWorkspaceDashboard


class ProjectWorkspaceTab(QWidget):
    """Independent project/workspace surface with durable flow overview."""

    resume_requested = Signal(str)

    def __init__(self, page_manager: PageManagerTab, parent=None):
        super().__init__(parent)
        self._page_manager = page_manager
        root = wrap_in_card(self)
        surface = QWidget()
        layout = QVBoxLayout(surface)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # Keep the legacy workspace/project controls fully alive for existing
        # callbacks and automation, but move them out of the visual surface.
        # The redesigned dashboard exposes the same actions through a compact
        # project menu so the home page matches the supplied mockup.
        self._legacy_project_bar = page_manager._build_project_workspace_bar()
        self._legacy_project_bar.setVisible(False)
        layout.addWidget(self._legacy_project_bar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self._dashboard = ProjectWorkspaceDashboard(page_manager)
        self._dashboard.resume_requested.connect(self.resume_requested.emit)
        scroll.setWidget(self._dashboard)
        layout.addWidget(scroll, 1)
        root.addWidget(surface, 1)

        page_manager.project_changed.connect(self._on_project_changed)

    def set_workspace_active(self, active: bool) -> None:
        self._dashboard.set_workspace_active(active)

    def refresh_project_state(self) -> None:
        self._dashboard.request_refresh()

    def set_runtime_task(self, task) -> None:
        self._dashboard.set_runtime_task(task)

    def resume_current_stage(self) -> None:
        self._dashboard.resume_current_stage()

    def _on_project_changed(self, _context) -> None:
        self._dashboard.request_refresh()
