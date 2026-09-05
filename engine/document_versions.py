# -*- coding: utf-8 -*-
"""OCR / replacement / AI three-version project workspace."""
from __future__ import annotations
import copy, json, time, uuid, hashlib, difflib, re, unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from models.document import UnifiedDocument, Block, BlockType
from utils.atomic_io import atomic_write_text, atomic_write_bytes



def document_identity(doc: UnifiedDocument | None) -> str:
    """Stable identity for separating consecutive books in one GUI session.

    Formatter/replacement versions keep the same page image paths and block IDs,
    while a newly imported book changes them.  The digest intentionally ignores
    mutable OCR text so normal processing does not look like a new book.
    """
    if doc is None:
        return ""
    page_paths = [str(Path(p.image_path).expanduser()) for p in doc.pages if p.image_path]
    if page_paths:
        payload = "pages\0" + "\0".join(page_paths)
    else:
        block_ids = [str(b.id) for b in doc.blocks[:32] if getattr(b, "id", "")]
        payload = "blocks\0" + "\0".join(block_ids)
        if not block_ids:
            payload += "\0" + str(doc.metadata.title or "") + "\0" + str(len(doc.blocks))
    return hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()


def document_content_fingerprint(doc: UnifiedDocument | None) -> str:
    """Stable digest of the editable document content and structure.

    ``document_identity`` deliberately ignores text so Formatter/replacement
    revisions of the same book remain in one session.  Workspaces also need a
    second signal, however: an old replacement/AI result must not be presented
    as current after the upstream OCR/Formatter text has changed.  This digest
    excludes mutable bibliographic/export metadata and hashes only the ordered
    blocks that define the current textual/asset structure.
    """
    if doc is None:
        return ""
    digest = hashlib.sha256()
    for index, block in enumerate(getattr(doc, "blocks", []) or []):
        metadata = dict(getattr(block, "metadata", {}) or {})
        # Runtime/review annotations do not change the authoritative text base.
        consumed = bool(metadata.get("consumed"))
        fields = (
            index,
            str(getattr(block, "id", "") or ""),
            str(getattr(getattr(block, "type", None), "value", getattr(block, "type", "")) or ""),
            str(getattr(block, "text", "") or ""),
            int(getattr(block, "page_index", 0) or 0),
            int(getattr(block, "page_number", 0) or 0),
            int(getattr(block, "chapter_index", 0) or 0),
            int(getattr(block, "order_in_page", 0) or 0),
            consumed,
        )
        digest.update(json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()

IMAGE_TYPES = {BlockType.COVER, BlockType.COLOR_ILLUS, BlockType.ILLUSTRATION,
               BlockType.TITLE_PAGE, BlockType.FRONTISPIECE, BlockType.INSERT,
               BlockType.ADVERTISEMENT, BlockType.MAP_PAGE, BlockType.CHARACTER_SHEET,
               BlockType.IMAGE_REF, BlockType.BLANK}

def propagate_chapter_membership(doc: UnifiedDocument) -> int:
    """Assign every body block to the nearest preceding chapter heading.

    Older OCR/replacement documents often set ``chapter_index`` only on CHAPTER
    blocks.  AI batching then puts all body text into ``chapter_000`` and emits
    the real chapter headings as empty chapters.  Sequence order is authoritative:
    a heading starts a chapter and all following text/assets inherit it until the
    next heading.
    """
    current = 0
    next_index = 1
    changed = 0
    for block in doc.blocks:
        if block.type == BlockType.CHAPTER:
            existing = int(getattr(block, "chapter_index", 0) or 0)
            current = existing if existing >= next_index else next_index
            next_index = current + 1
            if block.chapter_index != current:
                block.chapter_index = current
                changed += 1
            continue
        if current and block.chapter_index != current:
            block.chapter_index = current
            changed += 1
    if changed:
        doc.add_log("chapter_membership", f"按章节标题为 {changed} 个内容块补全章节归属", changed)
    return changed


def _compact_ai_guard_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    value = re.sub(r"[\s　]+", "", value)
    return value.translate(str.maketrans({
        "『": "「", "』": "」", "“": "「", "”": "」",
        "—": "ー", "―": "ー", "−": "ー", "ｰ": "ー",
    }))

@dataclass
class VersionInfo:
    type: str = "ocr"
    parent: str = ""
    created_at: str = ""
    revision: int = 1
    changes: list[dict] = field(default_factory=list)

class DocumentVersionStore:
    """Keeps three independent complete UnifiedDocument snapshots on disk or in memory."""
    FILES = {"ocr": "ocr_document.json", "replacement": "replacement_result.json", "ai": "ai_result.json"}
    def __init__(self, project_dir: str | None = None):
        self.project_dir = Path(project_dir) if project_dir else None
        self.documents: dict[str, UnifiedDocument] = {}
        self.info: dict[str, VersionInfo] = {}
        if self.project_dir:
            for d in ("documents", "revisions/ocr_history", "revisions/replacement_history",
                      "revisions/ai_history", "assets/cover", "assets/images", "logs", "exports"):
                (self.project_dir / d).mkdir(parents=True, exist_ok=True)

    def set(self, kind: str, doc: UnifiedDocument, parent: str = "", changes=None, autosave=True):
        if kind not in self.FILES: raise ValueError(f"Unknown document version: {kind}")
        cloned = copy.deepcopy(doc)
        old = self.info.get(kind)
        rev = (old.revision + 1) if old else 1
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.documents[kind] = cloned
        self.info[kind] = VersionInfo(kind, parent, stamp, rev, list(changes or []))
        cloned.add_log("document_version", f"saved {kind} revision {rev}; parent={parent or '-'}", 0)
        if autosave and self.project_dir: self.save(kind)
        return cloned

    def get(self, kind: str) -> Optional[UnifiedDocument]:
        doc = self.documents.get(kind)
        return copy.deepcopy(doc) if doc else None

    def clear(self, kind: str, delete_disk: bool = True):
        """Clear one version without affecting the other two versions."""
        if kind not in self.FILES:
            raise ValueError(f"Unknown document version: {kind}")
        self.documents.pop(kind, None)
        self.info.pop(kind, None)
        if delete_disk and self.project_dir:
            target = self.project_dir / "documents" / self.FILES[kind]
            target.unlink(missing_ok=True)
            if kind == "ai":
                (self.project_dir / "logs" / "ai_changes.json").unlink(missing_ok=True)
                (self.project_dir / "exports" / "ai_typeset.css").unlink(missing_ok=True)

    def save(self, kind: str):
        if not self.project_dir or kind not in self.documents: return
        target = self.project_dir / "documents" / self.FILES[kind]
        payload = self.documents[kind].to_dict()
        payload["version"] = self.info[kind].__dict__
        if target.exists():
            h = self.project_dir / "revisions" / f"{kind}_history" / f"{int(time.time())}_{target.name}"
            atomic_write_bytes(h, target.read_bytes())
        atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        if kind == "ai":
            atomic_write_text(
                self.project_dir / "logs" / "ai_changes.json",
                json.dumps(self.info[kind].changes, ensure_ascii=False, indent=2) + "\n",
            )
            css = str(getattr(self.documents[kind].metadata, "ai_epub_css", "") or "").strip()
            css_path = self.project_dir / "exports" / "ai_typeset.css"
            if css:
                atomic_write_text(css_path, css)
            else:
                css_path.unlink(missing_ok=True)

    @classmethod
    def open(cls, project_dir: str):
        store = cls(project_dir)
        for kind, name in cls.FILES.items():
            p = store.project_dir / "documents" / name
            if p.exists():
                raw = json.loads(p.read_text(encoding="utf-8")); v = raw.pop("version", {})
                store.documents[kind] = UnifiedDocument.from_dict(raw)
                store.info[kind] = VersionInfo(**{k: v.get(k, getattr(VersionInfo(), k)) for k in VersionInfo.__dataclass_fields__})
        return store


def validate_ai_document(source: UnifiedDocument, result: UnifiedDocument, require_complete: bool = False) -> list[str]:
    errors=[]
    if not result.blocks or not any(b.text.strip() for b in result.blocks if b.type not in IMAGE_TYPES):
        errors.append("AI result has no body text")
    src_ch={b.chapter_index for b in source.blocks if b.chapter_index}
    dst_ch={b.chapter_index for b in result.blocks if b.chapter_index}
    if src_ch and not src_ch.issubset(dst_ch):
        errors.append("chapters were lost")
    if len(result.pages) != len(source.pages):
        errors.append("page structure changed")
    src_imgs=[(p.page_no,p.page_type.value,p.image_path) for p in source.pages if p.image_path]
    dst_imgs=[(p.page_no,p.page_type.value,p.image_path) for p in result.pages if p.image_path]
    if src_imgs != dst_imgs:
        errors.append("image resources or paths changed")
    src_chars=sum(len(b.text) for b in source.blocks if b.type not in IMAGE_TYPES and not (b.metadata or {}).get("consumed"))
    dst_chars=sum(len(b.text) for b in result.blocks if b.type not in IMAGE_TYPES and not (b.metadata or {}).get("consumed"))
    if src_chars and dst_chars < src_chars * .72:
        errors.append("abnormal large-scale deletion")

    if require_complete:
        # Every non-empty source block must remain traceable through source_block_ids.
        # This catches a model silently dropping one paragraph even when whole-book
        # character totals still look acceptable.
        required = {
            b.id: b for b in source.blocks
            if b.type not in IMAGE_TYPES and (b.text or "").strip() and not (b.metadata or {}).get("consumed")
        }
        outputs_by_source: dict[str, list[str]] = {}
        for block in result.blocks:
            if block.type in IMAGE_TYPES or not (block.text or "").strip():
                continue
            ids = [str(x) for x in ((block.metadata or {}).get("source_block_ids") or [])]
            for sid in ids:
                outputs_by_source.setdefault(sid, []).append(block.text)
        missing = [sid for sid in required if sid not in outputs_by_source]
        if missing:
            errors.append(f"{len(missing)} source blocks were omitted")

        lost_content = 0
        for sid, source_block in required.items():
            source_text = _compact_ai_guard_text(source_block.text)
            if len(source_text) < 40 or sid not in outputs_by_source:
                continue
            output_text = _compact_ai_guard_text("".join(outputs_by_source[sid]))
            if not output_text:
                lost_content += 1
                continue
            matcher = difflib.SequenceMatcher(None, source_text, output_text, autojunk=False)
            common = sum(m.size for m in matcher.get_matching_blocks() if m.size)
            coverage = common / max(len(source_text), 1)
            if coverage < 0.42:
                lost_content += 1
        if lost_content:
            errors.append(f"{lost_content} source blocks lost substantial content")
    return errors


def build_ai_document(source: UnifiedDocument, payload: dict) -> tuple[UnifiedDocument,list[dict]]:
    """Build a full AI document while preserving the AI chapter/block order.

    Text must never be re-sorted by inherited OCR page numbers: split/merged AI blocks can
    legitimately inherit incomplete page metadata, and page-based sorting can move later
    prose in front of the first chapter heading. Page/order metadata is retained only for
    asset anchoring and diagnostics.
    """
    propagate_chapter_membership(source)
    out = copy.deepcopy(source)
    source_by_id = {b.id: b for b in source.blocks}
    source_rank = {b.id: i for i, b in enumerate(source.blocks)}
    new_text: list[Block] = []
    changes = payload.get("changes", []) or []
    chapters = payload.get("chapters")
    if not isinstance(chapters, list):
        raise ValueError("AI response missing chapters[]")

    global_order = 0
    for ci, ch in enumerate(chapters, 1):
        chapter_id = str(ch.get("id") or f"chapter_{ci:03d}")
        for order, item in enumerate(ch.get("blocks", []) or []):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            ids = [str(x) for x in (item.get("source_block_ids", []) or [])]
            valid_ids = [x for x in ids if x in source_by_id]
            if not valid_ids:
                raise ValueError(
                    f"AI block {item.get('id') or global_order + 1} has no valid source_block_ids; "
                    "orphan text was rejected to prevent it from appearing before a chapter title"
                )
            template = source_by_id[valid_ids[0]]
            b = copy.deepcopy(template)
            try:
                b.type = BlockType(item.get("type", b.type.value))
            except Exception:
                pass
            b.id = str(item.get("id") or f"ai_{uuid.uuid4().hex}")
            b.text = text
            b.modified_by = "ai_full_document"
            b.chapter_index = template.chapter_index if template.chapter_index else ci
            b.metadata = dict(b.metadata or {})
            b.metadata["source_block_ids"] = valid_ids
            b.metadata["ai_chapter_id"] = chapter_id
            b.metadata["ai_order"] = order
            b.metadata["ai_global_order"] = global_order
            b.metadata["source_rank"] = min(source_rank[x] for x in valid_ids)
            new_text.append(b)
            global_order += 1

    # Preserve AI text order exactly. Insert project assets near the first AI block whose
    # source rank follows the asset's original position. Assets never determine text order.
    ordered: list[Block] = list(new_text)
    assets = [copy.deepcopy(b) for b in source.blocks if b.type in IMAGE_TYPES]
    for asset in assets:
        rank = source_rank.get(asset.id, len(source.blocks))
        insert_at = len(ordered)
        for i, block in enumerate(ordered):
            if block.type in IMAGE_TYPES:
                continue
            if int((block.metadata or {}).get("source_rank", len(source.blocks))) > rank:
                insert_at = i
                break
        ordered.insert(insert_at, asset)

    out.blocks = ordered
    out.add_log("ai_full_document", f"AI generated {len(new_text)} editable blocks in AI order", len(changes))
    errs = validate_ai_document(source, out, require_complete=bool(payload.get("complete_document")))
    if errs:
        raise ValueError("AI result validation failed: " + "; ".join(errs))
    # AI may merge/split source blocks and can inherit stale Ruby metadata from
    # the first template block. Rebuild the locked side-channel from source
    # lineage after the authoritative text validation has succeeded.
    try:
        from adapters.findtext_centernet_ruby import carry_ruby_overlay, has_ruby_overlay
        if has_ruby_overlay(source):
            carry_ruby_overlay(source, out)
    except Exception:
        pass
    return out, changes


def cleanup_ai_covered_fragments(source: UnifiedDocument, result: UnifiedDocument) -> int:
    """Consume an AI block when the previous AI block already absorbed its source.

    Typeset mode may correctly repair a broken OCR boundary by completing the first
    block, while the differential protocol preserves the unchanged continuation block.
    That creates visible duplicates such as ``...魔王」といった。`` followed by
    ``魔王といった。``.  We only consume the second block when both conditions hold:

    1. its text is strongly covered by the end of the previous AI block; and
    2. the previous AI block closely matches the *joined original source blocks*.

    The second condition protects deliberate repeated lines: if the source really had
    two complete repetitions, one AI block cannot closely match their concatenation.
    """
    source_by_id = {b.id: b for b in source.blocks}

    def compact(text: str) -> str:
        value = unicodedata.normalize("NFKC", text or "")
        value = re.sub(r"[\s　]+", "", value)
        return value.translate(str.maketrans({
            "『": "「", "』": "」", "“": "「", "”": "」",
            "—": "ー", "―": "ー", "−": "ー", "ｰ": "ー",
        }))

    def source_text(block: Block) -> str:
        ids = [str(x) for x in ((block.metadata or {}).get("source_block_ids") or [])]
        return "".join(source_by_id[x].text for x in ids if x in source_by_id)

    def covered_by_previous(previous_text: str, current_text: str) -> bool:
        left, right = compact(previous_text), compact(current_text)
        if not left or not right or len(right) < 4 or len(right) > 64 or len(right) >= len(left):
            return False
        if left.endswith(right) or right in left[-max(len(right) + 20, 80):]:
            return True
        tail = left[-max(len(right) + 24, 80):]
        matcher = difflib.SequenceMatcher(None, right, tail, autojunk=False)
        matches = [m for m in matcher.get_matching_blocks() if m.size]
        anchored = max(
            (m.size for m in matches if m.a + m.size >= len(right) - 2 and m.b + m.size >= len(tail) - 2),
            default=0,
        )
        coverage = sum(m.size for m in matches) / max(len(right), 1)
        return anchored >= max(4, int(len(right) * 0.45)) and coverage >= 0.68

    changed = 0
    previous_index = None
    for index, block in enumerate(result.blocks):
        if block.type in IMAGE_TYPES:
            previous_index = None
            continue
        if not (block.text or "").strip() or (block.metadata or {}).get("consumed"):
            continue
        if previous_index is None:
            previous_index = index
            continue

        previous = result.blocks[previous_index]
        if previous.chapter_index != block.chapter_index:
            previous_index = index
            continue
        if not covered_by_previous(previous.text, block.text):
            previous_index = index
            continue

        src_prev = compact(source_text(previous))
        src_cur = compact(source_text(block))
        ai_prev = compact(previous.text)
        joined = src_prev + src_cur
        if not src_prev or not src_cur or not joined:
            previous_index = index
            continue

        joined_ratio = difflib.SequenceMatcher(None, ai_prev, joined, autojunk=False).ratio()
        length_ratio = min(len(ai_prev), len(joined)) / max(len(ai_prev), len(joined), 1)
        # A completed previous block should resemble the two original OCR pieces
        # together.  Deliberate repeated source lines produce a much poorer ratio.
        if joined_ratio < 0.70 or length_ratio < 0.68:
            previous_index = index
            continue

        original = block.text
        block.ocr_raw = block.ocr_raw or original
        block.text = ""
        block.modified_by = "ai_covered_fragment"
        block.metadata = {
            **(block.metadata or {}),
            "consumed": True,
            "consumed_by": previous.id,
            "consumed_reason": "ai_previous_block_absorbed_source_continuation",
        }
        changed += 1
        # Keep previous_index unchanged so multiple tiny tails can be absorbed.

    if changed:
        result.add_log("ai_covered_fragment_cleanup", f"AI排版后清理 {changed} 个被前块覆盖的续接残片", changed)
        # A consumed tail may contain Ruby whose base text was absorbed by the
        # previous AI block. Reattach from the immutable source after cleanup.
        try:
            from adapters.findtext_centernet_ruby import carry_ruby_overlay, has_ruby_overlay
            if has_ruby_overlay(source):
                carry_ruby_overlay(source, result)
        except Exception:
            pass
    return changed


def ai_request_payload(doc: UnifiedDocument) -> dict:
    propagate_chapter_membership(doc)
    images=[]
    for b in doc.blocks:
        if b.type in IMAGE_TYPES and (b.image_path or b.type != BlockType.BLANK):
            images.append({"image_id":b.id,"original_path":b.image_path,"anchor_block_id":b.image_anchor,
                           "placement":(b.metadata or {}).get("placement","after"),"page":b.page,
                           "role":b.type.value})
    blocks=[]
    for b in doc.blocks:
        if b.type in IMAGE_TYPES:
            continue
        # Consumed placeholders preserve OCR page/image anchoring but are not real
        # prose. Sending them to AI can resurrect a tail fragment that replacement
        # already removed. Empty blocks are likewise structural placeholders only.
        if (b.metadata or {}).get("consumed") or not (b.text or "").strip():
            continue
        blocks.append({"id":b.id,"type":b.type.value,"text":b.text,"chapter_id":f"chapter_{b.chapter_index:03d}",
                       "page":b.page,"reading_order":b.reading_order,"image_before":[],"image_after":[],"protected":False})
    return {"document_id":"project","source_version":"replacement","metadata":doc.metadata.to_dict(),"blocks":blocks,"images":images,
            "rules":{"allow_merge":True,"allow_split":True,"fix_punctuation":True,"fix_ocr":True,
                     "complete_obvious_missing_characters":True,"forbid_style_rewrite":True,"forbid_plot_expansion":True}}
