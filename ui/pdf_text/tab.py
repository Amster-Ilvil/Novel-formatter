from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtWidgets import QFrame, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QPlainTextEdit, QProgressBar, QFileDialog, QMessageBox, QSizePolicy, QCheckBox
from PySide6.QtCore import Qt, Signal

from utils.async_generation import GenerationGuard
from ui.responsive import preserve_button_text
from ui.dialogs import show_error_dialog
from ui.common.signals import WorkerSignals
from ui.common.toast import notify
from ui.common.styling import ACC, ACC_BG, BORDER, CARD, INK, MUTED, TONAL, LIGHT_LOG_STYLE, accent_button, wrap_in_card
from ui.design.metrics import (
    PDF_PAGE_MARGIN_X, PDF_PAGE_MARGIN_TOP, PDF_PAGE_MARGIN_BOTTOM,
    PDF_LEFT_WIDTH, PDF_COLUMN_GAP, PDF_RUN_HEIGHT,
)

class PdfTextLayerTab(QWidget):
    """
    直接读取带文字层的 PDF（不调用任何 OCR 模型）。PDF 来源统一由 Page
    Manager 管理，本页不再提供第二个“导入 PDF”入口；页面管理中的封面、
    插图、空白页、版权页等已确认类型可直接作为提取覆盖层。提取完的结果
    和 OCR 结果走同一个信号出口，接入 Formatter / EPUB Builder。
    """
    doc_extracted = Signal(object)
    page_manager_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pending_pdf: str | None = None
        self._extract_generation = GenerationGuard()
        self._extract_running = False
        self._page_manager_context_provider = None
        self._page_manager_context: dict = {}
        self._last_extracted_doc = None
        self._last_extracted_pdf: str | None = None
        self._build()

    def _build(self):
        root = wrap_in_card(self)
        root.setContentsMargins(
            PDF_PAGE_MARGIN_X, PDF_PAGE_MARGIN_TOP,
            PDF_PAGE_MARGIN_X, PDF_PAGE_MARGIN_BOTTOM,
        )
        root.setSpacing(PDF_COLUMN_GAP)
        card_css = (f"QFrame#pdfCard{{background:{CARD};border:1px solid {BORDER};border-radius:16px;}}")

        left = QWidget()
        left.setFixedWidth(PDF_LEFT_WIDTH)
        left.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        left_outer = QVBoxLayout(left)
        left_outer.setContentsMargins(0, 0, 0, 0)
        left_outer.setSpacing(16)

        # 卡片一：输入
        inp = QFrame()
        inp.setObjectName("pdfCard")
        inp.setStyleSheet(card_css)
        inp.setFixedHeight(176)
        il = QVBoxLayout(inp)
        il.setContentsMargins(20, 18, 20, 20)
        il.setSpacing(10)
        hdr = QLabel("PDF 文字层直读")
        hdr.setStyleSheet(f"color:{INK};font-size:15px;font-weight:750;background:none;border:none;")
        il.addWidget(hdr)
        sub = QLabel("PDF 来源由页面管理统一提供；本页不再重复导入文件。可自动识别上下双页，不调用任何 OCR 模型。")
        sub.setStyleSheet(f"color:{MUTED};font-size:11px;background:none;border:none;")
        sub.setWordWrap(True)
        il.addWidget(sub)
        source_cap = QLabel("当前页面管理 PDF")
        source_cap.setStyleSheet(f"color:{MUTED};font-size:10px;font-weight:650;background:none;border:none;")
        il.addWidget(source_cap)
        self._input_lbl = QLabel("尚未在页面管理载入 PDF")
        self._input_lbl.setStyleSheet(f"color:{INK};font-size:12px;font-weight:650;background:none;border:none;")
        self._input_lbl.setWordWrap(True)
        il.addWidget(self._input_lbl)
        hint = QLabel("如需更换文件，请回到页面管理重新导入 PDF。")
        hint.setStyleSheet(f"color:{MUTED};font-size:10px;background:none;border:none;")
        hint.setWordWrap(True)
        il.addWidget(hint)
        left_outer.addWidget(inp)

        # 卡片二：自动规则（说明性文字，不影响提取逻辑）
        rules = QFrame()
        rules.setObjectName("pdfCard")
        rules.setStyleSheet(card_css)
        rules.setFixedHeight(228)
        rl_ = QVBoxLayout(rules)
        rl_.setContentsMargins(20, 16, 20, 16)
        rl_.setSpacing(8)
        cap = QLabel("自动规则")
        cap.setStyleSheet(f"color:{MUTED};font-size:11px;font-weight:650;letter-spacing:1px;background:none;border:none;")
        rl_.addWidget(cap)
        for text in ("按字号自动跳过振假名", "按位置自动跳过页码", "不调用任何 OCR 模型"):
            row = QLabel(f"✓   {text}")
            row.setStyleSheet(f"color:{INK};font-size:12px;background:none;border:none;")
            rl_.addWidget(row)
        self._stacked_pages_check = QCheckBox("自动识别上下双页文字层")
        self._stacked_pages_check.setChecked(False)
        self._stacked_pages_check.setToolTip(
            "检测 PDF 页面中央的全宽空白带；命中后按上半→下半作为两个逻辑页提取。\n"
            "不会固定从正中间硬切；如遇特殊 PDF 可取消勾选。"
        )
        self._stacked_pages_check.setStyleSheet(
            f"QCheckBox{{color:{INK};font-size:12px;background:none;border:none;}}"
        )
        rl_.addWidget(self._stacked_pages_check)

        self._page_manager_check = QCheckBox("使用页面管理已确认页类型（非正文跳过）")
        self._page_manager_check.setChecked(False)
        self._page_manager_check.setToolTip(
            "仅使用页面管理里由用户亲自确认过的页类型；导入时自动填充的“正文”建议不会参与。\n"
            "当前 PDF 与页面管理来源一致时，封面/插图/空白页/目录/后记/版权页等会直接跳过文字层提取。"
        )
        self._page_manager_check.setStyleSheet(
            f"QCheckBox{{color:{INK};font-size:12px;background:none;border:none;}}"
        )
        self._page_manager_check.toggled.connect(lambda _checked: self._refresh_page_manager_context())
        rl_.addWidget(self._page_manager_check)

        pm_row = QHBoxLayout()
        pm_row.setContentsMargins(0, 0, 0, 0)
        pm_row.setSpacing(8)
        self._page_manager_status = QLabel("页面管理：未绑定")
        self._page_manager_status.setWordWrap(True)
        self._page_manager_status.setStyleSheet(
            f"color:{MUTED};font-size:10px;background:none;border:none;"
        )
        pm_row.addWidget(self._page_manager_status, 1)
        self._page_manager_btn = QPushButton("去标记")
        self._page_manager_btn.setToolTip("打开页面管理；选中封面、插图、空白页等后直接设置类型")
        preserve_button_text(self._page_manager_btn)
        self._page_manager_btn.clicked.connect(self.page_manager_requested.emit)
        self._page_manager_btn.setStyleSheet(
            f"QPushButton{{color:{ACC};background:{ACC_BG};border:1px solid {BORDER};"
            "border-radius:8px;padding:4px 9px;font-size:10px;font-weight:650;}"
            f"QPushButton:hover{{background:{TONAL};}}"
        )
        pm_row.addWidget(self._page_manager_btn, 0)
        rl_.addLayout(pm_row)
        left_outer.addWidget(rules)
        left_outer.addStretch(1)

        self._progress_lbl = QLabel("")
        self._progress_lbl.setStyleSheet(f"color:{MUTED};font-size:11px;")
        self._progress_lbl.setVisible(False)
        left_outer.addWidget(self._progress_lbl)
        self._run_btn = accent_button("▶  开始提取")
        self._run_btn.setFixedHeight(PDF_RUN_HEIGHT)
        preserve_button_text(self._run_btn)
        # preserve_button_text() intentionally defaults to Maximum for toolbars;
        # the PDF golden master uses a full-width primary action.
        self._run_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # App-level QSS may otherwise collapse a custom fixed height back to the
        # generic 34px button metric.  Keep the reference 42px action stable.
        self._run_btn.setStyleSheet(
            self._run_btn.styleSheet()
            + f"\nQPushButton{{min-height:{PDF_RUN_HEIGHT-2}px;max-height:{PDF_RUN_HEIGHT-2}px;"
              "padding-top:0;padding-bottom:0;}"
        )
        self._run_btn.clicked.connect(self._run_extract)
        left_outer.addWidget(self._run_btn)
        self._review_btn = QPushButton("导出损坏字 GPT 字形复核包")
        self._review_btn.setEnabled(False)
        self._review_btn.setToolTip(
            "只导出 PDF 文字层中 U+FFFD 损坏字形；同一内嵌字体 + glyph id 默认只复核一次，"
            "不会把整本图片重新送去识别。"
        )
        preserve_button_text(self._review_btn)
        self._review_btn.clicked.connect(self._export_visual_review_pack)
        left_outer.addWidget(self._review_btn)
        self._repair_import_btn = QPushButton("导入 GPT 修复 JSON")
        self._repair_import_btn.setEnabled(False)
        self._repair_import_btn.setToolTip(
            "导入复核包对应的 corrections JSON；只允许替换原来明确为 U+FFFD 的位置。"
        )
        preserve_button_text(self._repair_import_btn)
        self._repair_import_btn.clicked.connect(self._import_visual_repairs)
        left_outer.addWidget(self._repair_import_btn)
        root.addWidget(left, 0)

        right = QFrame()
        right.setObjectName("pdfCard")
        right.setStyleSheet(card_css)
        right.setMinimumWidth(520)
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(20, 16, 20, 18)
        rl.setSpacing(10)
        log_hdr = QLabel("提取日志")
        log_hdr.setStyleSheet(f"color:{INK};font-size:14px;font-weight:700;background:none;border:none;")
        rl.addWidget(log_hdr)
        self._prog = QProgressBar()
        self._prog.setFixedHeight(8)
        self._prog.setTextVisible(False)
        self._prog.setVisible(False)
        rl.addWidget(self._prog)
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setStyleSheet(LIGHT_LOG_STYLE)
        rl.addWidget(self._log_view, 1)
        root.addWidget(right, 1)


    def set_page_manager_context_provider(self, provider) -> None:
        """Attach a read-only Page Manager context provider.

        The provider accepts ``pdf_path`` (or ``None`` for the current Page
        Manager source) and returns a JSON-safe dictionary containing:
        ``matched``, ``source_pdf``, ``page_overrides`` and a short ``reason``.
        This deliberately keeps PDF text extraction independent from Page
        Manager widgets while letting both paths share one explicit page-type
        authority layer.
        """
        self._page_manager_context_provider = provider
        self._refresh_page_manager_context()

    def notify_page_manager_context_changed(self) -> None:
        """Refresh the lightweight binding badge after Page Manager edits."""
        self._refresh_page_manager_context()

    def _resolve_page_manager_context(self, pdf_path: str | None = None) -> dict:
        provider = self._page_manager_context_provider
        if not callable(provider):
            return {
                "matched": False,
                "page_overrides": {},
                "reason": "页面管理未连接",
            }
        try:
            result = dict(provider(pdf_path) or {})
        except Exception as exc:
            return {
                "matched": False,
                "page_overrides": {},
                "reason": f"页面管理状态读取失败：{exc}",
            }
        result.setdefault("matched", False)
        result.setdefault("page_overrides", {})
        result.setdefault("reason", "")
        return result

    def _refresh_page_manager_context(self) -> dict:
        # PDF Text Layer no longer owns a file picker.  Always resolve the
        # current Page Manager source so switching books cannot leave a stale
        # path cached in this workspace.
        context = self._resolve_page_manager_context(None)
        self._page_manager_context = context
        source_pdf = str(context.get("source_pdf") or "").strip()
        if bool(context.get("matched")) and source_pdf:
            self._pending_pdf = source_pdf
            if hasattr(self, "_input_lbl"):
                self._input_lbl.setText(Path(source_pdf).name)
        else:
            self._pending_pdf = None
            if hasattr(self, "_input_lbl"):
                if source_pdf:
                    self._input_lbl.setText(f"{Path(source_pdf).name} · 当前页面映射不可直接使用")
                else:
                    self._input_lbl.setText("尚未在页面管理载入 PDF")
        if not hasattr(self, "_page_manager_status"):
            return context
        if not self._page_manager_check.isChecked():
            if bool(context.get("matched")):
                self._page_manager_status.setText("页面类型标记：已关闭；仍使用页面管理当前 PDF 作为来源")
            else:
                reason = str(context.get("reason") or "页面管理没有可用的单一 PDF 来源")
                self._page_manager_status.setText(f"页面管理：无法开始 · {reason}")
            return context
        if bool(context.get("matched")):
            confirmed = int(context.get("confirmed_count", len(context.get("page_overrides") or {})) or 0)
            skipped = int(context.get("skip_count", 0) or 0)
            page_count = int(context.get("page_count", 0) or 0)
            name = Path(source_pdf).name if source_pdf else "当前 PDF"
            if confirmed:
                self._page_manager_status.setText(
                    f"页面管理：已绑定 {name} · {page_count} 页 · 已确认 {confirmed} 页 · 将跳过 {skipped} 个非正文页"
                )
            else:
                self._page_manager_status.setText(
                    f"页面管理：已绑定 {name} · {page_count} 页 · 尚未人工确认页类型"
                )
        else:
            reason = str(context.get("reason") or "页面管理没有可用的单一 PDF 来源")
            self._page_manager_status.setText(f"页面管理：无法开始 · {reason}")
        return context

    def reset_for_new_book(self):
        """Detach the PDF extraction workspace from a previous book/session."""
        self._extract_generation.invalidate()
        self._extract_running = False
        self._pending_pdf = None
        self._last_extracted_doc = None
        self._last_extracted_pdf = None
        self._run_btn.setEnabled(True)
        if hasattr(self, "_review_btn"):
            self._review_btn.setEnabled(False)
        if hasattr(self, "_repair_import_btn"):
            self._repair_import_btn.setEnabled(False)
        self._input_lbl.setText("尚未在页面管理载入 PDF")
        self._page_manager_context = {}
        self._refresh_page_manager_context()
        self._prog.setVisible(False)
        self._progress_lbl.setVisible(False)
        self._log_view.clear()

    def shutdown_cleanup(self) -> None:
        self._extract_generation.invalidate()
        self._extract_running = False

    def _extract_is_current(self, token: int) -> bool:
        return self._extract_generation.is_current(token)

    def _run_extract(self):
        if self._extract_running:
            return
        context = self._refresh_page_manager_context()
        if not bool(context.get("matched")) or not self._pending_pdf:
            reason = str(context.get("reason") or "页面管理没有可用的单一 PDF 来源")
            QMessageBox.warning(
                self, "请先在页面管理导入 PDF",
                "PDF 文字层不再单独导入文件。请先在页面管理导入一个 PDF，"
                "并保持其物理页映射可与原 PDF 对应。\n\n" + reason,
            )
            return
        pdf_path = self._pending_pdf
        # Resolve once more against the concrete source before starting so a
        # late Page Manager change cannot silently switch page-type overlays.
        context = self._resolve_page_manager_context(pdf_path)
        if not bool(context.get("matched")):
            QMessageBox.warning(
                self, "页面管理 PDF 已变化",
                str(context.get("reason") or "当前页面管理来源已变化，请重新确认后再开始。"),
            )
            self._refresh_page_manager_context()
            return
        page_manager_enabled = bool(self._page_manager_check.isChecked())
        context_snapshot = dict(context)
        page_overrides = {}
        if page_manager_enabled and bool(context_snapshot.get("matched")):
            page_overrides = {int(k): str(v) for k, v in dict(context_snapshot.get("page_overrides") or {}).items()}
        split_mode = "auto" if self._stacked_pages_check.isChecked() else "off"
        token = self._extract_generation.begin()
        self._extract_running = True

        self._log_view.clear()
        if page_overrides:
            skipped = sorted(page_no for page_no, page_type in page_overrides.items() if page_type != "paragraph")
            preview = "、".join(str(v) for v in skipped[:18])
            if len(skipped) > 18:
                preview += "…"
            self._log_view.appendPlainText(
                f"📑 页面管理：应用 {len(page_overrides)} 个已确认页类型；非正文跳过 {len(skipped)} 页"
                + (f"（{preview}）" if preview else "")
            )
        elif page_manager_enabled:
            reason = str(context_snapshot.get("reason") or "未找到与当前 PDF 对应的页面管理标记")
            self._log_view.appendPlainText(f"📑 页面管理：未应用标记 · {reason}")
        else:
            self._log_view.appendPlainText("📑 页面管理：本次已关闭")
        self._run_btn.setEnabled(False)
        self._prog.setVisible(True)
        self._prog.setRange(0, 1)
        self._prog.setValue(0)
        self._progress_lbl.setVisible(True)
        self._progress_lbl.setText("准备中…")

        def on_progress(current, total, label):
            signals.progress.emit(current, total)
            signals.log.emit(f"  [{current:3d}/{total}] {label}")

        def worker():
            try:
                from adapters.pdf_text_layer import has_text_layer, extract_pdf_text_layer
                if not has_text_layer(pdf_path):
                    raise ValueError("该 PDF 没有可提取的文字层（可能是扫描版 PDF），请改用 OCR 适配器")
                doc = extract_pdf_text_layer(
                    pdf_path, page_overrides=page_overrides, verbose=False,
                    progress_callback=on_progress, split_stacked_pages=split_mode,
                )
                doc.metadata.pdf_text_page_manager_report = {
                    "enabled": page_manager_enabled,
                    "matched": bool(context_snapshot.get("matched")),
                    "source_pdf": str(context_snapshot.get("source_pdf") or pdf_path),
                    "confirmed_count": int(context_snapshot.get("confirmed_count", len(page_overrides)) or 0),
                    "skip_count": int(context_snapshot.get("skip_count", sum(1 for t in page_overrides.values() if t != "paragraph")) or 0),
                    "page_overrides": {str(k): v for k, v in sorted(page_overrides.items())},
                    "reason": str(context_snapshot.get("reason") or ""),
                }
                signals.finished.emit(doc)
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        signals = self._signals = WorkerSignals()
        signals.finished.connect(lambda doc, t=token: self._on_done(doc, t))
        signals.error.connect(lambda msg, t=token: self._on_error(msg, t))
        signals.log.connect(lambda line, t=token: self._log_view.appendPlainText(line) if self._extract_is_current(t) else None)
        signals.progress.connect(lambda current, total, t=token: self._on_progress(current, total, t))
        threading.Thread(target=worker, daemon=True).start()

    def _export_visual_review_pack(self):
        doc = self._last_extracted_doc
        pdf_path = str(self._last_extracted_pdf or "").strip()
        if doc is None or not pdf_path:
            notify(self, "请先完成一次 PDF 文字层提取。", "warning")
            return
        replacement_count = int(getattr(doc.metadata, "pdf_text_replacement_glyph_count", 0) or 0)
        if replacement_count <= 0:
            notify(self, "当前 PDF 文字层没有检测到 U+FFFD 损坏字形。", "info")
            return
        default_name = str(Path(pdf_path).with_name(Path(pdf_path).stem + "_PDF文字层_GPT复核包.zip"))
        output, _ = QFileDialog.getSaveFileName(self, "导出 GPT 视觉复核包", default_name, "ZIP (*.zip)")
        if not output:
            return
        if not output.lower().endswith(".zip"):
            output += ".zip"
        self._review_btn.setEnabled(False)
        self._review_btn.setText("正在导出复核包…")
        self._log_view.appendPlainText(f"\n🔎 正在导出 {replacement_count} 个损坏字形的局部视觉复核包…")

        signals = WorkerSignals()
        self._review_signals = signals

        def worker():
            try:
                from tools.pdf_text_visual_review_pack import build_visual_review_pack
                result = build_visual_review_pack(pdf_path, doc, output)
                signals.finished.emit(result)
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        def done(result):
            self._review_btn.setText("导出损坏字 GPT 字形复核包")
            self._review_btn.setEnabled(True)
            result = dict(result or {})
            count = int(result.get("items", 0) or 0)
            units = int(result.get("review_units", count) or count)
            mode = str(result.get("mode", "per_item") or "per_item")
            if mode == "glyph_cluster":
                detail = f"{count} 个损坏位置 → {units} 个字形组"
            else:
                detail = f"{count} 个损坏字形"
            self._log_view.appendPlainText(f"✅ GPT 复核包已导出：{detail} · {output}")
            notify(
                self,
                f"已导出 {detail}。\nGPT 只需要看红框局部字形，不必重新识别整本。",
                "success",
            )

        def failed(message):
            self._review_btn.setText("导出损坏字 GPT 字形复核包")
            self._review_btn.setEnabled(True)
            show_error_dialog(self, "GPT 复核包导出失败", str(message))

        signals.finished.connect(done)
        signals.error.connect(failed)
        threading.Thread(target=worker, daemon=True).start()

    def _import_visual_repairs(self):
        if self._last_extracted_doc is None:
            notify(self, "请先完成一次 PDF 文字层提取。", "warning")
            return
        path, _ = QFileDialog.getOpenFileName(self, "导入 GPT 修复 JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            import json
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON 必须是对象，并包含 corrections 或 glyph_corrections")
            from core.pdf_text_visual_review import apply_pdf_text_visual_review_payload
            repaired = apply_pdf_text_visual_review_payload(
                self._last_extracted_doc, payload, copy_document=True
            )
            report = dict(getattr(repaired.metadata, "pdf_text_visual_repair_report", {}) or {})
            applied = int(report.get("applied", 0) or 0)
            skipped = int(report.get("skipped", 0) or 0)
            remaining = int(report.get("remaining", 0) or 0)
            if applied <= 0:
                QMessageBox.warning(
                    self, "没有应用修复",
                    f"没有找到可安全应用的损坏字形。跳过 {skipped} 项。\n"
                    "请确认 JSON 来自当前这次 PDF 复核包，并且每项只填写一个字符。"
                )
                return
            self._last_extracted_doc = repaired
            self._review_btn.setEnabled(bool(remaining and self._last_extracted_pdf))
            self._repair_import_btn.setEnabled(bool(remaining))
            self._log_view.appendPlainText(
                f"✅ GPT 视觉修复已导入：应用 {applied} 项，跳过 {skipped} 项，仍剩 {remaining} 个损坏字形。"
            )
            self.doc_extracted.emit(repaired)
            notify(
                self,
                f"已安全替换 {applied} 个原始 U+FFFD 位置；仍剩 {remaining} 个。\n"
                "原始文字层保留在 ocr_raw，修复稿已重新发送到后续排版/EPUB 流程。",
                "success",
            )
        except Exception as exc:
            show_error_dialog(self, "GPT 修复 JSON 导入失败", str(exc))

    def _on_progress(self, current, total, token: int):
        if not self._extract_is_current(token):
            return
        self._prog.setRange(0, max(total, 1))
        self._prog.setValue(current)
        self._progress_lbl.setText(f"{current} / {total} 页")

    def _finish_extract(self, token: int) -> bool:
        if not self._extract_is_current(token):
            return False
        self._extract_running = False
        self._run_btn.setEnabled(True)
        self._prog.setVisible(False)
        self._progress_lbl.setVisible(False)
        return True

    def _on_done(self, doc, token: int):
        if not self._finish_extract(token):
            return
        logical_pages = int(getattr(doc.metadata, "pdf_text_logical_page_count", len(doc.pages)) or len(doc.pages))
        physical_pages = int(getattr(doc.metadata, "pdf_text_physical_page_count", logical_pages) or logical_pages)
        stacked_pages = int(getattr(doc.metadata, "pdf_text_stacked_page_count", 0) or 0)
        skipped_pages = len(getattr(doc.metadata, "pdf_text_skipped_physical_pages", []) or [])
        replacement_count = int(getattr(doc.metadata, "pdf_text_replacement_glyph_count", 0) or 0)
        replacement_rate = float(getattr(doc.metadata, "pdf_text_replacement_glyph_rate", 0.0) or 0.0)
        self._last_extracted_doc = doc
        self._last_extracted_pdf = str(self._pending_pdf or "") or None
        self._review_btn.setEnabled(bool(replacement_count and self._last_extracted_pdf))
        self._repair_import_btn.setEnabled(bool(replacement_count))
        self._log_view.appendPlainText(
            f"\n✅ 完成: {physical_pages} 物理页 → {logical_pages} 逻辑页；"
            f"自动拆分 {stacked_pages} 页；页面管理跳过 {skipped_pages} 页；"
            f"{len(doc.blocks)} 个块，{len(doc.toc)} 章节"
        )
        if replacement_count:
            self._log_view.appendPlainText(
                f"⚠ 原 PDF 文字层损坏：检测到 {replacement_count} 个替换字形 "
                f"({replacement_rate:.2%})；已保留原页坐标，可仅对这些位置做 GPT/OCR 视觉复核。"
            )
        self.doc_extracted.emit(doc)

    def _on_error(self, msg, token: int):
        if not self._finish_extract(token):
            return
        show_error_dialog(self, "提取失败", msg)

