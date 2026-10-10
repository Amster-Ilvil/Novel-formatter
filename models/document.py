#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Document Model（统一文档模型）
所有 OCR 适配器的输出、Formatter Engine 的输入/输出都使用此结构。

版本历史存储（v2）：
    不再把每一步的完整 JSON 快照都堆在内存里的 history 列表中（旧方案：
    每跑一步 Pipeline，就把当前状态整个序列化一次塞进 list，一本几百页
    的书跑完 9 个步骤，内存里会同时存在 9 份几乎重复的全量 JSON）。
    现在改用内容寻址存储（做法类似 Git）：
        - 每次 commit 只是把当前内容序列化后按内容 SHA-1 存成一个 blob 文件，
          内容完全相同的两次 commit 会自动复用同一个 blob（天然去重）。
        - commit 对象只记录 {root blob id, parent commit id, step, message}，
          本身也按内容寻址存储。
        - 历史链条持久化在磁盘上的仓库目录里，不是纯内存对象，App 重启后
          仍能通过 repo_path 找回（旧方案每次重开 App 历史就没了）。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import hashlib
import json
try:
    import orjson as _orjson
except ImportError:  # optional acceleration; standard json remains supported
    _orjson = None
import time
import uuid
from pathlib import Path



class BlockType(str, Enum):
    # 页面级别（Page Manager 分类结果）
    COVER            = "cover"            # 封面
    COLOR_ILLUS      = "color_illus"      # 彩色插图
    BLANK            = "blank"            # 空白页
    TOC_PAGE         = "toc_page"         # 目录页（原始图片）
    ILLUSTRATION     = "illustration"     # 黑白插图
    AFTERWORD        = "afterword"        # 后记页
    COLOPHON         = "colophon"         # 版权页/奥付
    HALF_ILLUS       = "half_illustration"# 半页插图
    TITLE_PAGE       = "title_page"       # 扉页/书名页
    FRONTISPIECE     = "frontispiece"     # 卷首插画
    INSERT           = "insert"           # 插页
    ADVERTISEMENT    = "advertisement"    # 广告页
    INDEX_PAGE       = "index"            # 索引
    APPENDIX         = "appendix"         # 附录
    MAP_PAGE         = "map"              # 地图页
    CHARACTER_SHEET  = "character_sheet"  # 人物设定页
    UNKNOWN          = "unknown"          # 未分类

    # 文本块级别（Formatter Engine 处理结果）
    PARAGRAPH     = "paragraph"      # 普通段落
    DIALOGUE      = "dialogue"       # 对白「」
    CHAPTER       = "chapter"        # 章节标题
    SECTION       = "section"        # 小节标题（幕間等）
    RUBY          = "ruby"           # 带振假名文本
    FOOTNOTE      = "footnote"       # 脚注
    TOC_ENTRY     = "toc_entry"      # 目录条目
    IMAGE_REF     = "image_ref"      # 图片引用（内联）
    HEADER_FOOTER = "header_footer"  # 已识别的页眉页脚（待过滤）


@dataclass
class BoundingBox:
    """归一化坐标 [0,1]，原点左上角"""
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0

    @classmethod
    def from_pixels(cls, x1, y1, x2, y2, page_w, page_h) -> BoundingBox:
        return cls(
            x=x1 / page_w, y=y1 / page_h,
            w=(x2 - x1) / page_w, h=(y2 - y1) / page_h
        )

    @classmethod
    def from_apple_vision(cls, box: dict) -> BoundingBox:
        """Convert Apple Vision lower-left normalized coordinates to our upper-left system."""
        x = float(box.get("x", 0) or 0)
        y = float(box.get("y", 0) or 0)
        w = float(box.get("w", box.get("width", 0)) or 0)
        h = float(box.get("h", box.get("height", 0)) or 0)
        return cls(
            x=max(0.0, min(1.0, x)),
            y=max(0.0, min(1.0, 1.0 - y - h)),
            w=max(0.0, min(1.0, w)),
            h=max(0.0, min(1.0, h)),
        )


@dataclass
class Block:
    """文档中的一个内容块"""
    type: BlockType
    text: str = ""

    # 位置信息
    page: int = 0
    bbox: Optional[BoundingBox] = None
    reading_order: int = 0
    page_index: Optional[int] = None
    page_number: Optional[int] = None
    order_in_page: Optional[int] = None
    text_direction: Optional[str] = None
    source_format: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    # 质量信息
    confidence: float = 1.0
    ocr_raw: str = ""           # OCR 原始文字，用于 diff 高亮
    modified_by: str = ""       # 哪个 Formatter 步骤修改了此块

    # 图片引用
    image_path: str = ""        # 图片文件路径（type=IMAGE_REF 时使用）
    image_anchor: str = ""      # 锚点，记录此图应插在哪段正文之后

    # 章节 / TOC
    chapter_index: int = 0      # 0=非章节块，≥1=第几章

    def to_dict(self) -> dict:
        d = {
            "type": self.type.value,
            "text": self.text,
            "page": self.page,
            "reading_order": self.reading_order,
            "confidence": round(self.confidence, 4),
        }
        if self.page_index is not None:
            d["page_index"] = self.page_index
        if self.page_number is not None:
            d["page_number"] = self.page_number
        if self.order_in_page is not None:
            d["order_in_page"] = self.order_in_page
        if self.text_direction is not None:
            d["text_direction"] = self.text_direction
        if self.source_format is not None:
            d["source_format"] = self.source_format
        if self.metadata:
            d["metadata"] = self.metadata
        if self.id:
            d["id"] = self.id
        if self.bbox:
            d["bbox"] = {"x": self.bbox.x, "y": self.bbox.y,
                         "w": self.bbox.w, "h": self.bbox.h}
        if self.image_path:
            d["image_path"] = self.image_path
        if self.image_anchor:
            d["image_anchor"] = self.image_anchor
        if self.ocr_raw and self.ocr_raw != self.text:
            d["ocr_raw"] = self.ocr_raw
        if self.modified_by:
            d["modified_by"] = self.modified_by
        if self.chapter_index:
            d["chapter_index"] = self.chapter_index
        return d


@dataclass
class PageInfo:
    """页面元信息（来自 Page Manager 分类结果）"""
    page_no: int
    page_type: BlockType       # cover / illustration / text / blank …
    image_path: str = ""
    width: int = 0
    height: int = 0
    confidence: float = 1.0    # 自动分类置信度

    def to_dict(self) -> dict:
        return {
            "page_no": self.page_no,
            "page_type": self.page_type.value,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class Metadata:
    title: str = ""
    author: str = ""
    publisher: str = ""
    series: str = ""
    volume: str = ""
    language: str = "ja"
    # OCR profile metadata keeps Japanese vertical and Simplified Chinese
    # horizontal runs isolated throughout save/load, comparison and export.
    ocr_mode: str = "ja_vertical"
    writing_direction: str = "vertical-rl"
    ocr_profile_version: int = 1
    formatter_profile: str = "ja_light_novel"
    # publication=preserve CJK glyph identity; compatibility=map Kangxi radicals for reader/search compatibility
    unicode_normalization_policy: str = "publication"
    isbn: str = ""
    description: str = ""
    source_engine: str = ""    # 使用的 OCR 引擎名称
    preserve_ocr_layout: bool = False  # 固定原 OCR 块/段落结构，跳过会重排段落的 Formatter 步骤
    pdf_text_layer_mode: bool = False  # PDF 可选文字层专用 Formatter；与普通图片 OCR 规则隔离
    pdf_keep_afterwords: bool = False  # 默认不保留作者前书/后记；仅在界面明确勾选时保留
    pdf_remove_generated_matter: bool = True  # PDFNovels/站点生成前后置页默认清理，与作者前后书独立
    pdf_restore_indents: bool = True  # PDF格式处理：出版段首缩进，默认开启
    pdf_format_processed: bool = False
    pdf_format_report: dict = field(default_factory=dict)
    pdf_text_source_char_counts: dict = field(default_factory=dict)  # 无损字符保全基线（忽略布局空白）
    pdf_text_source_chars: int = 0
    pdf_text_output_chars: int = 0
    pdf_text_missing_chars: int = 0
    pdf_text_extra_chars: int = 0
    pdf_text_character_guard_passed: bool = True
    pdf_text_guard_report: dict = field(default_factory=dict)
    # Selectable-PDF stacked-page provenance. One physical PDF page may contain
    # two logical pages arranged top/bottom. Keep the mapping explicit so
    # formatter/publication stages never need to infer it again.
    pdf_text_split_mode: str = ""  # auto | off | force
    pdf_text_physical_page_count: int = 0
    pdf_text_logical_page_count: int = 0
    pdf_text_stacked_page_count: int = 0
    pdf_text_page_map: dict = field(default_factory=dict)
    pdf_text_stacked_split_reports: dict = field(default_factory=dict)
    # Read-only bridge from Page Manager. Only explicitly confirmed labels may
    # suppress native PDF text extraction.
    pdf_text_page_overrides: dict = field(default_factory=dict)
    pdf_text_skipped_physical_pages: list = field(default_factory=list)
    pdf_text_page_manager_report: dict = field(default_factory=dict)
    pdf_text_furigana_chars_skipped: int = 0
    pdf_text_page_number_chars_skipped: int = 0
    pdf_text_page_number_columns_skipped: int = 0
    ai_processing_mode: str = ""  # correction | typeset
    ai_layout_locked: bool = False  # EPUB 必须优先使用 AI 返回的文本与段落结构
    ai_epub_css: str = ""  # AI 纠错排版返回的完整 EPUB CSS；仅 typeset 版本使用
    ai_epub_css_name: str = ""  # 保存到磁盘时使用的建议文件名
    ai_epub_css_source: str = ""  # ai | fallback，便于界面提示和诊断
    replacement_mode: str = ""  # strict_full | smart_patch | compare_only
    replacement_source_hash: str = ""  # 严格覆盖来源正文 SHA-256
    replacement_output_hash: str = ""  # 严格覆盖输出正文 SHA-256
    replacement_exact_match: bool = False  # 结构化正文是否与来源 100% 一致
    replacement_source_chars: int = 0
    replacement_output_chars: int = 0
    replacement_missing_chars: int = 0
    replacement_extra_chars: int = 0
    replacement_pending_images: int = 0
    replacement_literal_exact_match: bool = False
    replacement_layout_passed: bool = True
    replacement_overlong_blocks: int = 0
    replacement_mixed_dialogue_blocks: int = 0
    replacement_unbalanced_dialogue_blocks: int = 0
    replacement_reflowed_blocks: int = 0
    replacement_unresolved_layout_blocks: int = 0
    replacement_quote_repairs: int = 0
    authoritative_text: bool = False
    authoritative_report: dict = field(default_factory=dict)
    authoritative_draft: bool = False
    authoritative_indexed_protocol: int = 0
    authoritative_checkpoint_dir: str = ""
    ocr_repair_mode: str = ""  # readability | strict
    column_sentence_reflow_applied: bool = False  # OCR 页已在进入 Formatter 前完成逐列成句
    column_sentence_reflow_version: int = 0
    column_sentence_reflow_max_columns: int = 0
    column_ocr_audit: dict = field(default_factory=dict)  # per-page expected/recognized/model/DOCX column IDs
    # Durable, per-physical-column OCR input provenance.  Block metadata can be
    # projected/reflowed away later (especially page-role -> sentence projection),
    # so the exact prepared/actual input hashes also live at document level.
    # Values are JSON-safe dictionaries keyed by stable physical column_id.
    column_ocr_input_records: dict = field(default_factory=dict)
    column_ocr_integrity_passed: bool = False
    # Geometry-only Ruby candidates captured while normal OCR is already splitting columns.
    # They contain no recognized Ruby text and never participate in OCR voting/fusion.
    ruby_candidate_pages: dict = field(default_factory=dict)
    # Ruby preservation is a locked side-channel on authoritative output only.
    # Keep these fields explicit so generic JSON/repository round-trips cannot
    # silently discard them (older code stored them dynamically in __dict__).
    ruby_preservation_enabled: bool = False
    ruby_overlay_scope: str = ""
    ruby_input_contract: str = ""
    ruby_preservation_engine: str = ""
    ruby_preservation_report: dict = field(default_factory=dict)
    ruby_overlay_transfer_report: dict = field(default_factory=dict)
    ocr_review_report: dict = field(default_factory=dict)  # OCR + 手动输入疑点筛查汇总
    page_asset_sync_signature: str = ""  # Page Manager overlay signature; avoids redundant full-document sync
    # Standalone multimodal image-book workflow.  These fields are deliberately
    # metadata-only so traditional OCR/Formatter contracts remain untouched.
    ai_image_provider: str = ""
    ai_image_model: str = ""
    ai_image_pipeline_version: str = ""
    ai_image_report: dict = field(default_factory=dict)
    ai_translation_target: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}


@dataclass
class TocEntry:
    title: str
    chapter_index: int
    block_index: int           # 对应 UnifiedDocument.blocks 的索引

    def to_dict(self) -> dict:
        return {"title": self.title, "chapter_index": self.chapter_index,
                "block_index": self.block_index}


# ── 版本仓库：内容寻址存储（做法类似 Git 的 blob + commit）───────────────────

def _serialize_blob_bytes(data: dict) -> bytes:
    """Serialize repository content once, using an optional fast encoder.

    Large selectable-PDF documents can contain tens of thousands of blocks.
    Serializing the same snapshot twice with the stdlib encoder (once for the
    hash and again for writing) dominated the complete 4693-page workflow.
    ``orjson`` is used when already available, but remains optional so offline
    installs without it keep working through the standard-library fallback.
    """
    if _orjson is not None:
        return _orjson.dumps(data)
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _compute_blob_id(data: dict) -> str:
    """内容的 SHA-1 哈希。相同内容→相同 id，天然去重。"""
    return hashlib.sha1(_serialize_blob_bytes(data)).hexdigest()


class Repository:
    """
    文档版本仓库：用一个目录持久化存储所有 blob 和 commit。
    目录结构（有意仿照 .git）：
        objects/xx/yyyy...      —— 按 SHA-1 前两位分桶存放的 blob 文件
        refs/heads/<branch>     —— 分支名 → 最新 commit id
        HEAD                    —— 指向当前分支
    """

    def __init__(self, repo_path: str):
        self.path = Path(repo_path)
        self.objects_dir = self.path / "objects"
        self.refs_dir = self.path / "refs" / "heads"
        self.head_file = self.path / "HEAD"
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.refs_dir.mkdir(parents=True, exist_ok=True)

    def _blob_path(self, blob_id: str) -> Path:
        return self.objects_dir / blob_id[:2] / blob_id[2:]

    def store_blob(self, content: dict) -> str:
        """存储一份内容（文档快照或 commit 元数据），返回内容寻址 id。

        Serialize once and write those exact bytes.  This preserves content
        addressing while avoiding a second full JSON traversal for huge books.
        """
        payload = _serialize_blob_bytes(content)
        blob_id = hashlib.sha1(payload).hexdigest()
        blob_file = self._blob_path(blob_id)
        if not blob_file.exists():
            blob_file.parent.mkdir(parents=True, exist_ok=True)
            blob_file.write_bytes(payload)
        return blob_id

    def read_blob(self, blob_id: str) -> dict:
        blob_file = self._blob_path(blob_id)
        if not blob_file.exists():
            raise FileNotFoundError(f"Blob {blob_id} 不存在于仓库 {self.path}")
        return json.loads(blob_file.read_text(encoding="utf-8"))

    # commit 本身也是一个按内容寻址存储的 blob
    store_commit = store_blob
    read_commit = read_blob

    def write_ref(self, branch: str, commit_id: str):
        (self.refs_dir / branch).write_text(commit_id + "\n", encoding="utf-8")

    def read_ref(self, branch: str) -> Optional[str]:
        ref_file = self.refs_dir / branch
        return ref_file.read_text(encoding="utf-8").strip() if ref_file.exists() else None

    def set_head(self, branch: str):
        self.head_file.write_text(f"ref: refs/heads/{branch}\n", encoding="utf-8")

    def get_head_branch(self) -> str:
        if self.head_file.exists():
            content = self.head_file.read_text(encoding="utf-8").strip()
            if content.startswith("ref: refs/heads/"):
                return content[len("ref: refs/heads/"):]
        return "main"

    def get_head_commit(self) -> Optional[str]:
        return self.read_ref(self.get_head_branch())

    def log(self, commit_id: Optional[str], max_count: int = 1000) -> list[dict]:
        """从 commit_id 开始沿 parent 链一直往回走，返回 [{"id": ..., **commit_fields}, ...]（新→旧）"""
        commits = []
        cur = commit_id
        seen = set()
        while cur and cur not in seen and len(commits) < max_count:
            seen.add(cur)
            data = self.read_commit(cur)
            commits.append({"id": cur, **data})
            cur = data.get("parent")
        return commits


@dataclass
class UnifiedDocument:
    """
    统一文档模型 —— 所有适配器输出、Formatter 处理结果的共同载体。

    不可变数据流原则（见 SDS 第十三章）：
        任何模块都不应该原地修改 UnifiedDocument，而是基于当前版本产生新版本。
        Formatter 的每个步骤内部都用 deepcopy 保证这一点；run_pipeline() 会把
        每一步的结果 commit 进仓库，使 Undo/Redo 和前后 Diff 成为可能。

    版本历史通过 repo（Repository）+ commit_id 追踪，而不是内存里的 list：
        doc.commit(repo_path, "step_name")   → 保存当前状态为新 commit
        doc.commit_log()                     → 列出从当前 commit 往回的历史
        doc.rollback_to_commit(commit_id)     → 还原到某个 commit（返回新对象）

    使用方式：
        doc = UnifiedDocument()
        doc.metadata.title = "魔法科高校の劣等生"
        doc.blocks.append(Block(...))
        print(doc.to_json())
    """
    metadata: Metadata = field(default_factory=Metadata)
    pages: list[PageInfo] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    toc: list[TocEntry] = field(default_factory=list)

    # 处理日志，记录每个 Formatter 步骤做了什么（人类可读的一行摘要）
    processing_log: list[dict] = field(default_factory=list)

    # 版本控制：仓库 + 当前所在的 commit（均可为 None，表示还没有提交过）
    repo: Optional[Repository] = None
    commit_id: Optional[str] = None

    def add_log(self, step: str, message: str, count: int = 0):
        self.processing_log.append({"step": step, "message": message, "count": count})

    # ── 版本控制 ─────────────────────────────────────────────────────────────

    def commit(self, repo_path: str, step_label: str, message: str = "") -> str:
        """
        把当前状态保存为一个新 commit（内容寻址，天然去重），
        父提交是 self.commit_id（如果还没提交过就是仓库的根提交）。
        返回新的 commit id，同时更新 self.repo / self.commit_id。
        """
        if self.repo is None:
            self.repo = Repository(repo_path)

        root_blob_id = self.repo.store_blob(self.to_dict())
        commit_data = {
            "root": root_blob_id,
            "parent": self.commit_id,
            "step": step_label,
            "message": message,
            "timestamp": time.time(),
        }
        commit_id = self.repo.store_commit(commit_data)

        branch = self.repo.get_head_branch()
        self.repo.write_ref(branch, commit_id)
        self.repo.set_head(branch)
        self.commit_id = commit_id
        return commit_id

    def commit_log(self, max_count: int = 1000) -> list[dict]:
        """返回从当前 commit 往回的历史，新→旧。没有仓库/未提交过则返回 []"""
        if self.repo is None or self.commit_id is None:
            return []
        return self.repo.log(self.commit_id, max_count=max_count)

    def rollback_to_commit(self, commit_id: str) -> UnifiedDocument:
        """把文档还原到指定 commit，返回一个新的 UnifiedDocument（不修改当前实例）"""
        if self.repo is None:
            raise RuntimeError("此文档还没有关联仓库，无法回滚")
        commit_data = self.repo.read_commit(commit_id)
        root = self.repo.read_blob(commit_data["root"])
        new_doc = UnifiedDocument.from_dict(root)
        new_doc.repo = self.repo
        new_doc.commit_id = commit_id
        return new_doc

    def text_blocks(self) -> list[Block]:
        """只返回正文类型的块"""
        TEXT_TYPES = {BlockType.PARAGRAPH, BlockType.DIALOGUE,
                      BlockType.CHAPTER, BlockType.SECTION, BlockType.RUBY}
        return [b for b in self.blocks if b.type in TEXT_TYPES and not (b.metadata or {}).get("consumed")]

    def image_blocks(self) -> list[Block]:
        return [b for b in self.blocks if b.type == BlockType.IMAGE_REF]

    def snapshot_clone(self) -> UnifiedDocument:
        """Return an exact, detached in-memory snapshot of this document.

        Generic ``copy.deepcopy`` spends most of its time recursively visiting the
        thousands of tiny dataclass/dict/list nodes in a book.  A document has a
        known typed shape, so clone the scalar/dataclass fields directly and deep
        copy only the JSON-shaped extension payloads in one batch.  This preserves
        exact floating-point confidence values (unlike a persistence round-trip,
        which intentionally rounds some fields) while keeping metadata, block
        metadata and processing logs fully independent.

        ``repo`` and ``commit_id`` are history pointers rather than content and are
        intentionally shared exactly like ``rollback_to_commit``/Formatter snapshots.
        """
        standard_fields = {
            "metadata", "pages", "blocks", "toc", "processing_log", "repo", "commit_id"
        }
        extras = {
            "metadata": self.metadata.__dict__,
            "block_metadata": [block.metadata for block in self.blocks],
            "processing_log": self.processing_log,
            # Preserve forward-compatible document-level extension fields too.
            # TEI import, for example, attaches ``tei_import_report`` directly to
            # UnifiedDocument rather than Metadata.
            "document_extras": {
                key: value for key, value in self.__dict__.items()
                if key not in standard_fields
            },
        }
        try:
            if _orjson is not None:
                detached = _orjson.loads(_orjson.dumps(extras))
            else:
                detached = json.loads(json.dumps(extras, ensure_ascii=False))
        except Exception:
            # Dynamic extension fields are expected to be JSON-safe because they
            # are persisted by ``to_dict``.  Keep a defensive fallback for third-
            # party adapters that temporarily attach richer Python objects.
            import copy as _copy
            detached = _copy.deepcopy(extras)

        metadata = Metadata()
        for key, value in dict(detached.get("metadata") or {}).items():
            if isinstance(key, str) and key:
                setattr(metadata, key, value)

        pages = [
            PageInfo(
                page_no=page.page_no, page_type=page.page_type,
                image_path=page.image_path, width=page.width, height=page.height,
                confidence=page.confidence,
            )
            for page in self.pages
        ]
        block_metadata = list(detached.get("block_metadata") or [])
        blocks: list[Block] = []
        for index, block in enumerate(self.blocks):
            bbox = None
            if block.bbox is not None:
                bbox = BoundingBox(block.bbox.x, block.bbox.y, block.bbox.w, block.bbox.h)
            metadata_value = (
                block_metadata[index]
                if index < len(block_metadata) and isinstance(block_metadata[index], dict)
                else {}
            )
            blocks.append(Block(
                type=block.type, text=block.text, page=block.page, bbox=bbox,
                reading_order=block.reading_order, page_index=block.page_index,
                page_number=block.page_number, order_in_page=block.order_in_page,
                text_direction=block.text_direction, source_format=block.source_format,
                metadata=metadata_value, id=block.id, confidence=block.confidence,
                ocr_raw=block.ocr_raw, modified_by=block.modified_by,
                image_path=block.image_path, image_anchor=block.image_anchor,
                chapter_index=block.chapter_index,
            ))
        toc = [TocEntry(item.title, item.chapter_index, item.block_index) for item in self.toc]
        cloned = UnifiedDocument(
            metadata=metadata, pages=pages, blocks=blocks, toc=toc,
            processing_log=list(detached.get("processing_log") or []),
            repo=self.repo, commit_id=self.commit_id,
        )
        for key, value in dict(detached.get("document_extras") or {}).items():
            if isinstance(key, str) and key and key not in standard_fields:
                setattr(cloned, key, value)
        return cloned

    def to_dict(self) -> dict:
        """
        内容快照，不含版本控制元信息（repo/commit_id 是"指针"，不是内容的一部分——
        两次内容完全相同的 commit 应该复用同一个 blob，如果把 commit_id 混进内容
        里算哈希，内容寻址去重就永远不会命中）。
        """
        return {
            "metadata": self.metadata.to_dict(),
            "pages": [p.to_dict() for p in self.pages],
            "blocks": [b.to_dict() for b in self.blocks],
            "toc": [t.to_dict() for t in self.toc],
            "processing_log": self.processing_log,
        }

    def to_json(self, indent: int = 2) -> str:
        """
        序列化为 JSON，供保存到文件 / GUI 载入。
        如果文档关联了仓库，额外写入 repo_path/commit_id，
        这样从 JSON 载入后仍能继续访问完整的版本历史（Undo 不会断）。
        """
        d = self.to_dict()
        if self.repo is not None:
            d["_repo_path"] = str(self.repo.path)
            d["_commit_id"] = self.commit_id
        return json.dumps(d, ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> UnifiedDocument:
        doc = cls()
        m = d.get("metadata", {})
        # Metadata.to_dict() intentionally serializes dynamic extension fields
        # from ``__dict__`` (multi-OCR roles/transports, future adapter audit
        # fields, etc.).  Older restore code only assigned fields predeclared on
        # the dataclass and therefore silently discarded those extensions after
        # a workspace restart.  Metadata is not slotted, so preserve every
        # serialized field for a lossless/forward-compatible round trip.
        for k, v in m.items():
            if isinstance(k, str) and k:
                setattr(doc.metadata, k, v)
        for p in d.get("pages", []):
            doc.pages.append(PageInfo(
                page_no=p["page_no"],
                page_type=BlockType(p["page_type"]),
                image_path=p.get("image_path", ""),
                width=p.get("width", 0),
                height=p.get("height", 0),
                confidence=p.get("confidence", 1.0),
            ))
        for b in d.get("blocks", []):
            bbox = None
            if "bbox" in b:
                bx = b["bbox"]
                bbox = BoundingBox(bx["x"], bx["y"], bx["w"], bx["h"])
            doc.blocks.append(Block(
                type=BlockType(b["type"]),
                text=b.get("text", ""),
                page=b.get("page", 0),
                bbox=bbox,
                reading_order=b.get("reading_order", 0),
                confidence=b.get("confidence", 1.0),
                ocr_raw=b.get("ocr_raw", ""),
                modified_by=b.get("modified_by", ""),
                image_path=b.get("image_path", ""),
                image_anchor=b.get("image_anchor", ""),
                chapter_index=b.get("chapter_index", 0),
                page_index=b.get("page_index"),
                page_number=b.get("page_number"),
                order_in_page=b.get("order_in_page"),
                text_direction=b.get("text_direction"),
                source_format=b.get("source_format"),
                metadata=b.get("metadata", {}),
                id=b.get("id", uuid.uuid4().hex),
            ))
        for t in d.get("toc", []):
            doc.toc.append(TocEntry(
                title=t["title"],
                chapter_index=t["chapter_index"],
                block_index=t["block_index"],
            ))
        doc.processing_log = d.get("processing_log", [])

        repo_path = d.get("_repo_path")
        if repo_path:
            doc.repo = Repository(repo_path)
            doc.commit_id = d.get("_commit_id")
        return doc

    @classmethod
    def from_json(cls, s: str) -> UnifiedDocument:
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_repo(cls, repo_path: str, branch: str = "main") -> UnifiedDocument:
        """从仓库中恢复某个分支最新的状态"""
        repo = Repository(repo_path)
        commit_id = repo.read_ref(branch)
        if not commit_id:
            raise ValueError(f"仓库 {repo_path} 里没有分支 {branch} 的提交记录")
        commit_data = repo.read_commit(commit_id)
        root = repo.read_blob(commit_data["root"])
        doc = cls.from_dict(root)
        doc.repo = repo
        doc.commit_id = commit_id
        return doc


def new_temp_repo_path() -> str:
    """Allocate a session-owned temporary history repository.

    Repositories remain available for Undo/Redo during the current application
    session, but are removed automatically on normal shutdown.  A crash leaves
    only an application-marked session directory that is reclaimed on the next
    launch.
    """
    from utils.session_temp import session_temp_registry

    return str(session_temp_registry().make_dir("document-repo"))
