from __future__ import annotations

from functools import partial
import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QTimer, QSize
from PySide6.QtWidgets import (
    QProgressBar,
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QListView, QSizePolicy, QToolButton, QMenu,
)

from core.project_flow import build_project_flow_snapshot
from ui.common.styling import (
    ACC_FILL, ACC_FILL_HOVER, ACC_TEXT,
    ACC, ACC_BG, BORDER, CARD, DANGER, INK, MUTED, SUBTLE, SUCCESS, SURFACE,
    HERO_A, HERO_B, ON_HERO, ON_HERO_TEXT, TONAL, TONAL_HOVER, ACC_HOVER,
)


_STAGE_LABELS = {
    "pages": "页面",
    "ocr": "OCR",
    "proof": "文字校对",
    "format": "格式处理",
    "export": "EPUB",
}

_STATE_STYLE = {
    "done": ("已完成", SUCCESS, "#E3F5EC"),
    "active": ("可继续", ACC_TEXT, ACC_BG),
    "optional": ("可选", "#6D4AE0", "#EFEAFF"),
    "stale": ("需更新", "#B7791F", "#FBF1DF"),
    "pending": ("未开始", MUTED, TONAL),
}


def _step_btn_style(active: bool) -> str:
    if active:
        return (f"QPushButton{{background:{ACC_FILL};color:#FFFFFF;border:none;border-radius:10px;padding:7px 12px;font-weight:700;}}"
                f"QPushButton:hover{{background:{ACC_FILL_HOVER};}}QPushButton:disabled{{background:{TONAL};color:{SUBTLE};}}")
    return (f"QPushButton{{background:{TONAL};color:{ACC_TEXT};border:none;border-radius:10px;padding:7px 12px;font-weight:600;}}"
            f"QPushButton:hover{{background:{TONAL_HOVER};}}QPushButton:disabled{{background:{TONAL};color:{SUBTLE};}}")


_HERO_BTN = (f"QPushButton{{background:{ON_HERO};color:{ON_HERO_TEXT};border:none;border-radius:12px;padding:11px 24px;font-size:13px;font-weight:700;}}"
             f"QPushButton:disabled{{background:rgba(255,255,255,90);color:rgba(255,255,255,170);}}")
_HERO_GHOST = ("QPushButton{background:rgba(255,255,255,46);color:#FFFFFF;border:1px solid rgba(255,255,255,90);"
               "border-radius:12px;padding:10px 18px;font-weight:600;}QPushButton:hover{background:rgba(255,255,255,80);}")


def _card(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("projectDashboardCard")
    frame.setStyleSheet(
        f"QFrame#projectDashboardCard{{background:{CARD};border:1px solid {BORDER};"
        "border-radius:14px;}"
    )
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(20, 18, 20, 18)
    layout.setSpacing(10)
    label = QLabel(title)
    label.setStyleSheet(f"color:{MUTED};font-size:11px;font-weight:650;letter-spacing:1px;")
    layout.addWidget(label)
    return frame, layout


class ProjectWorkspaceDashboard(QWidget):
    """Fast project home surface inspired by durable project-oriented editors.

    The dashboard never resolves large OCR payloads.  It displays only durable
    metadata and emits stable workspace keys when the user asks to continue.
    """

    resume_requested = Signal(str)

    def __init__(self, page_manager, parent=None):
        super().__init__(parent)
        self._page_manager = page_manager
        self._manager = page_manager.project_manager
        self._stage_widgets: dict[str, tuple[QLabel, QLabel, QPushButton]] = {}
        self._stage_nodes: dict[str, QLabel] = {}
        self._stage_boxes: dict[str, QFrame] = {}
        self._runtime_task = None
        self._project_tool_buttons: list[QPushButton] = []
        # Runtime progress and project signals can arrive in bursts.  Coalesce
        # dashboard disk reads in the same spirit as debounced dynamic-page
        # observers: invisible dashboards only become dirty; visible dashboards
        # perform at most one refresh per short burst.
        self._refresh_dirty = True
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(140)
        self._refresh_timer.timeout.connect(self._refresh_now)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 28)
        root.setSpacing(16)

        hero = QFrame()
        hero.setObjectName("projectHero")
        hero.setStyleSheet(
            f"QFrame#projectHero{{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 {HERO_A},stop:1 {HERO_B});"
            "border:none;border-radius:20px;}"
        )
        hero.setFixedHeight(201)
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(30, 26, 30, 24)
        hero_layout.setSpacing(16)
        cap = QLabel("当前项目")
        cap.setStyleSheet("color:rgba(255,255,255,200);font-size:11px;font-weight:650;letter-spacing:1px;background:none;")
        hero_layout.addWidget(cap)
        hero_row = QHBoxLayout()
        hero_row.setSpacing(14)
        hero_text = QVBoxLayout()
        hero_text.setSpacing(5)
        self._project_name = QLabel("未选择项目")
        self._project_name.setStyleSheet("color:#FFFFFF;font-size:28px;font-weight:750;background:none;")
        hero_text.addWidget(self._project_name)
        self._project_path = QLabel("创建或选择项目后，页面、OCR、裁决与 EPUB 会持续保存。")
        self._project_path.setWordWrap(True)
        self._project_path.setStyleSheet("color:rgba(255,255,255,210);font-size:11px;background:none;")
        hero_text.addWidget(self._project_path)
        self._last_activity = QLabel("")
        self._last_activity.setStyleSheet("color:rgba(255,255,255,180);font-size:11px;background:none;")
        hero_text.addWidget(self._last_activity)
        hero_row.addLayout(hero_text, 1)
        self._resume_btn = QPushButton("继续上次工作")
        self._resume_btn.setStyleSheet(_HERO_BTN)
        self._resume_btn.clicked.connect(self._resume)
        hero_row.addWidget(self._resume_btn, 0, Qt.AlignVCenter)
        refresh_btn = QPushButton("刷新")
        refresh_btn.setStyleSheet(_HERO_GHOST)
        refresh_btn.clicked.connect(self.refresh)
        hero_row.addWidget(refresh_btn, 0, Qt.AlignVCenter)
        project_menu_btn = QToolButton()
        project_menu_btn.setText("⋯")
        project_menu_btn.setPopupMode(QToolButton.InstantPopup)
        project_menu_btn.setToolTip("项目管理")
        project_menu_btn.setFixedSize(42, 40)
        project_menu_btn.setStyleSheet(_HERO_GHOST + "QToolButton{font-size:18px;padding:0;}QToolButton::menu-indicator{image:none;}")
        project_menu = QMenu(project_menu_btn)
        project_menu.addAction("新建项目", self._page_manager._new_project)
        project_menu.addAction("导入项目", self._page_manager._import_project_folder)
        project_menu.addAction("导入备份", self._page_manager._import_project_backup)
        project_menu.addSeparator()
        project_menu.addAction("选择工作区", self._page_manager._choose_workspace_root)
        project_menu.addAction("打开工作区", self._page_manager._open_workspace_root)
        project_menu_btn.setMenu(project_menu)
        hero_row.addWidget(project_menu_btn, 0, Qt.AlignVCenter)
        hero_layout.addLayout(hero_row)
        prog = QHBoxLayout()
        prog.setSpacing(12)
        self._progress = QProgressBar()
        self._progress.setRange(0, 5)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(6)
        self._progress.setStyleSheet(
            "QProgressBar{background:rgba(255,255,255,70);border:none;border-radius:3px;}"
            f"QProgressBar::chunk{{background:{ON_HERO};border-radius:3px;}}")
        prog.addWidget(self._progress, 1)
        self._progress_label = QLabel("0 / 5 步")
        self._progress_label.setStyleSheet("color:#FFFFFF;font-size:11px;font-weight:650;background:none;")
        prog.addWidget(self._progress_label)
        hero_layout.addLayout(prog)
        root.addWidget(hero)

        flow, flow_layout = _card("项目流程")
        flow.setFixedHeight(232)
        grid = QGridLayout()
        grid.setContentsMargins(0, 2, 0, 0)
        grid.setHorizontalSpacing(10)
        for index, key in enumerate(("pages", "ocr", "proof", "format", "export")):
            box = QFrame()
            box.setObjectName(f"projectFlow_{key}")
            self._stage_boxes[key] = box
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(14, 14, 14, 14)
            box_layout.setSpacing(8)
            top = QHBoxLayout()
            top.setSpacing(9)
            node = QLabel(str(index + 1))
            node.setFixedSize(26, 26)
            node.setAlignment(Qt.AlignCenter)
            top.addWidget(node)
            self._stage_nodes[key] = node
            title = QLabel(_STAGE_LABELS[key])
            title.setStyleSheet(f"color:{INK};font-size:14px;font-weight:700;background:none;")
            top.addWidget(title, 1)
            box_layout.addLayout(top)
            badge = QLabel("未开始")
            badge.setAlignment(Qt.AlignCenter)
            box_layout.addWidget(badge, 0, Qt.AlignLeft)
            detail = QLabel("—")
            detail.setWordWrap(True)
            detail.setMinimumHeight(44)
            detail.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            detail.setStyleSheet(f"color:{MUTED};font-size:11px;background:none;")
            box_layout.addWidget(detail, 1)
            open_btn = QPushButton("打开 →")
            open_btn.setStyleSheet(_step_btn_style(False))
            open_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            open_btn.clicked.connect(partial(self._open_stage, key))
            box_layout.addWidget(open_btn)
            grid.addWidget(box, 0, index)
            self._stage_widgets[key] = (badge, detail, open_btn)
        for column in range(5):
            grid.setColumnStretch(column, 1)
        flow_layout.addLayout(grid)
        root.addWidget(flow)

        lower = QHBoxLayout()
        lower.setSpacing(14)
        recent, recent_layout = _card("最近项目")
        recent.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        recent.setMinimumHeight(300)
        recent_toolbar = QHBoxLayout()
        recent_toolbar.setContentsMargins(0, 0, 0, 0)
        recent_toolbar.addStretch(1)
        self._recent_grid_btn = QPushButton("缩略图")
        self._recent_grid_btn.setCheckable(True)
        self._recent_grid_btn.setChecked(True)
        self._recent_list_btn = QPushButton("列表")
        self._recent_list_btn.setCheckable(True)
        self._recent_grid_btn.clicked.connect(lambda: self._set_recent_view_mode("grid"))
        self._recent_list_btn.clicked.connect(lambda: self._set_recent_view_mode("list"))
        recent_toolbar.addWidget(self._recent_grid_btn)
        recent_toolbar.addWidget(self._recent_list_btn)
        recent_layout.addLayout(recent_toolbar)
        self._recent = QListWidget()
        self._recent.setMinimumHeight(210)
        self._recent.setAlternatingRowColors(False)
        self._recent.setSelectionMode(QListWidget.SingleSelection)
        self._recent.setResizeMode(QListView.Adjust)
        self._recent.setMovement(QListView.Static)
        self._recent.itemDoubleClicked.connect(self._open_recent_item)
        recent_layout.addWidget(self._recent, 1)
        hint = QLabel("双击项目即可切换；缩略图来自项目持久化页面，不重新读取 OCR。")
        hint.setStyleSheet(f"color:{SUBTLE};font-size:10px;")
        hint.setWordWrap(True)
        recent_layout.addWidget(hint)
        lower.addWidget(recent, 3)

        tools, tools_layout = _card("项目工具")
        tools.setMinimumHeight(300)
        tools.setMaximumWidth(430)
        tools.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        tool_grid = QGridLayout()
        tool_grid.setHorizontalSpacing(8)
        tool_grid.setVerticalSpacing(8)
        actions = [
            ("备份项目", self._page_manager._export_project_backup),
            ("运行日志", self._page_manager._open_project_run_logs),
            ("检查项目", self._page_manager._audit_active_project),
            ("优化存储", self._page_manager._optimize_active_project_storage),
            ("项目目录", self._page_manager._open_active_project_folder),
        ]
        for idx, (text, callback) in enumerate(actions):
            btn = QPushButton(text)
            btn.clicked.connect(callback)
            self._project_tool_buttons.append(btn)
            tool_grid.addWidget(btn, idx // 2, idx % 2)
            if text == "优化存储":
                # Existing worker completion code owns this button state.
                self._page_manager._storage_opt_btn = btn
        delete_btn = QPushButton("删除项目")
        delete_btn.setProperty("role", "danger")
        delete_btn.clicked.connect(self._page_manager._delete_active_project)
        self._project_tool_buttons.append(delete_btn)
        tool_grid.addWidget(delete_btn, 2, 1)
        tools_layout.addLayout(tool_grid)
        tools_layout.addStretch(1)
        lower.addWidget(tools, 2)
        root.addLayout(lower, 1)
        self._recent_view_mode = "grid"
        self._set_recent_view_mode("grid")

        self.refresh()

    def set_workspace_active(self, active: bool) -> None:
        if active:
            self.request_refresh(immediate=True)

    def set_runtime_task(self, task) -> None:
        self._runtime_task = task
        self.request_refresh()

    def request_refresh(self, *, immediate: bool = False) -> None:
        self._refresh_dirty = True
        if not self.isVisible():
            return
        if immediate:
            self._refresh_timer.stop()
            self._refresh_now()
            return
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def refresh(self) -> None:
        """Force an immediate user-requested refresh."""
        self._refresh_dirty = True
        self._refresh_timer.stop()
        self._refresh_now()

    def _refresh_now(self) -> None:
        if not self._refresh_dirty:
            return
        self._refresh_dirty = False
        snapshot = build_project_flow_snapshot(self._manager, runtime_task=self._runtime_task)
        self._snapshot = snapshot
        active = bool(snapshot.project_path)
        self._project_name.setText(snapshot.project_name or "未选择项目")
        self._project_path.setText(
            snapshot.project_path
            if active
            else "创建或选择项目后，页面、OCR、裁决与 EPUB 会持续保存。"
        )
        if active and snapshot.last_run_stage:
            if snapshot.runtime_task_status in {"running", "cancelling"}:
                status_text = "运行中" if snapshot.runtime_task_status == "running" else "正在停止"
                self._last_activity.setText(f"当前任务：{snapshot.last_run_stage} · {status_text}")
            elif snapshot.resumable_ocr:
                self._last_activity.setText(
                    f"上次 OCR 可恢复 · 已记录 {snapshot.resume_completed_steps} 个完成阶段"
                )
            else:
                when = snapshot.last_run_finished_at or "—"
                self._last_activity.setText(
                    f"最近任务：{snapshot.last_run_stage} · {snapshot.last_run_status or 'unknown'} · {when}"
                )
        elif active:
            self._last_activity.setText("项目已建立，尚无运行记录。")
        else:
            self._last_activity.setText("")
        self._resume_btn.setEnabled(active)
        for button in self._project_tool_buttons:
            button.setEnabled(active)
        self._resume_btn.setText(snapshot.resume_reason if active else "继续上次工作")
        self._resume_btn.setProperty("resumeWorkspace", snapshot.resume_workspace)

        by_key = {stage.key: stage for stage in snapshot.stages}
        done = sum(1 for stage in snapshot.stages if stage.state == "done")
        self._progress.setValue(done)
        self._progress_label.setText(f"{done} / 5 步")
        for key, (badge, detail, button) in self._stage_widgets.items():
            stage = by_key.get(key)
            if stage is None:
                continue
            label, color, background = _STATE_STYLE.get(stage.state, _STATE_STYLE["pending"])
            badge.setText(label)
            self._style_stage(key, stage.state, button)
            badge.setStyleSheet(
                f"background:{background};color:{color};border-radius:8px;"
                "padding:2px 8px;font-size:10px;font-weight:650;"
            )
            detail.setText(stage.detail or "—")
            button.setProperty("workspaceKey", stage.workspace)
            button.setEnabled(active or key == "pages")

        self._recent.clear()
        active_path = snapshot.project_path
        for info in self._manager.list_projects()[:20]:
            marker = "● " if active_path and str(info.path) == active_path else ""
            updated = str(info.updated_at or "").replace("T", " ")[:16]
            item = QListWidgetItem(f"{marker}{info.name}\n{updated or info.path}")
            item.setData(Qt.UserRole, info.path)
            item.setToolTip(info.path)
            thumb = self._project_thumbnail_path(info.path)
            if thumb is not None:
                from PySide6.QtGui import QIcon
                item.setIcon(QIcon(str(thumb)))
            self._recent.addItem(item)
        if self._recent.count() == 0:
            item = QListWidgetItem("暂无项目")
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self._recent.addItem(item)
        self._set_recent_view_mode(getattr(self, "_recent_view_mode", "grid"), refresh_buttons=False)

    def _project_thumbnail_path(self, project_path: str | Path) -> Path | None:
        project = Path(project_path).expanduser()
        cover_dir = project / "assets" / "cover"
        if cover_dir.is_dir():
            for candidate in sorted(cover_dir.iterdir()):
                if candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and candidate.is_file():
                    return candidate
        state_path = project / "pages" / "state.json"
        try:
            raw = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        for value in list(raw.get("page_images") or []):
            candidate = Path(str(value)).expanduser()
            if not candidate.is_absolute():
                candidate = project / candidate
            if candidate.is_file():
                return candidate
        for folder in (project / "pages" / "processed", project / "pages" / "source"):
            if folder.is_dir():
                for candidate in sorted(folder.iterdir()):
                    if candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and candidate.is_file():
                        return candidate
        return None

    def _set_recent_view_mode(self, mode: str, *, refresh_buttons: bool = True) -> None:
        mode = "list" if str(mode) == "list" else "grid"
        self._recent_view_mode = mode
        if refresh_buttons:
            self._recent_grid_btn.setChecked(mode == "grid")
            self._recent_list_btn.setChecked(mode == "list")
        if mode == "grid":
            self._recent.setViewMode(QListView.IconMode)
            self._recent.setFlow(QListView.LeftToRight)
            self._recent.setWrapping(True)
            self._recent.setWordWrap(True)
            self._recent.setIconSize(QSize(104, 140))
            self._recent.setGridSize(QSize(190, 188))
            self._recent.setSpacing(8)
        else:
            self._recent.setViewMode(QListView.ListMode)
            self._recent.setFlow(QListView.TopToBottom)
            self._recent.setWrapping(False)
            self._recent.setWordWrap(False)
            self._recent.setIconSize(QSize(42, 56))
            self._recent.setGridSize(QSize())
            self._recent.setSpacing(4)

    def _style_stage(self, key: str, state: str, button: QPushButton) -> None:
        order = list(_STAGE_LABELS).index(key) + 1
        node, box = self._stage_nodes[key], self._stage_boxes[key]
        if state == "done":
            node.setText("✓")
            node.setStyleSheet(f"background:{SUCCESS};color:#FFFFFF;border-radius:13px;font-weight:700;")
        elif state == "active":
            node.setText(str(order))
            node.setStyleSheet(f"background:{ACC_FILL};color:#FFFFFF;border-radius:13px;font-weight:700;")
        else:
            node.setText(str(order))
            node.setStyleSheet(f"background:{CARD};color:{SUBTLE};border:1px solid {BORDER};border-radius:13px;")
        current = state == "active"
        box.setStyleSheet(
            f"QFrame#{box.objectName()}{{background:{ACC_BG if current else SURFACE};"
            f"border:1px solid {ACC if current else BORDER};border-radius:14px;}}")
        button.setStyleSheet(_step_btn_style(current))
        button.setText("继续 →" if current else "打开 →")

    def resume_current_stage(self) -> None:
        workspace = str(self._resume_btn.property("resumeWorkspace") or "book")
        self.resume_requested.emit(workspace)

    def _resume(self) -> None:
        self.resume_current_stage()

    def _open_stage(self, key: str) -> None:
        widgets = self._stage_widgets.get(str(key))
        if not widgets:
            return
        workspace = str(widgets[2].property("workspaceKey") or "book")
        self.resume_requested.emit(workspace)

    def _open_recent_item(self, item: QListWidgetItem) -> None:
        path = str(item.data(Qt.UserRole) or "")
        if not path:
            return
        self._page_manager._activate_project(path, restore_pages=True)
        self.refresh()


__all__ = ["ProjectWorkspaceDashboard"]
