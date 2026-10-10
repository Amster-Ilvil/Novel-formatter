#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TEI P5 interchange bridge for Novel Formatter.

This module intentionally does *not* replace the internal UnifiedDocument JSON
model.  TEI is used as an interoperable, inspectable interchange layer:

* ``facsimile/surface/zone`` preserves page geometry and links transcription
  blocks back to the source image regions through ``@facs``;
* ``standOff/listAnnotation`` keeps OCR/adjudication provenance outside the
  publication text so evidence can never leak into the reading text;
* ``xenoData`` embeds the canonical Novel Formatter snapshot as JSON for a
  lossless round trip when the file returns to this application.

The shape follows TEI P5 4.12.0 and patterns used by OCR-to-TEI projects such as
local-ocr and alto2tei.  The generated TEI remains useful to external tools even
when they ignore the Novel Formatter extension namespace.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable
import xml.etree.ElementTree as ET

from models.document import Block, BlockType, BoundingBox, PageInfo, TocEntry, UnifiedDocument
from utils.atomic_io import atomic_write_text

TEI_NS = "http://www.tei-c.org/ns/1.0"
NF_NS = "urn:novel-formatter:tei:1.0"
XML_NS = "http://www.w3.org/XML/1998/namespace"
TEI_VERSION = "4.12.0"
SNAPSHOT_SCHEMA = "novel-formatter-unified-document-v1"
SNAPSHOT_HASH = "sha256"
PROJECTION_VERSION = "2"

ET.register_namespace("", TEI_NS)
ET.register_namespace("nf", NF_NS)


def _tei(tag: str) -> str:
    return f"{{{TEI_NS}}}{tag}"


def _nf(tag: str) -> str:
    return f"{{{NF_NS}}}{tag}"


def _xml_attr(name: str) -> str:
    return f"{{{XML_NS}}}{name}"


def _safe_xml_id(value: str, prefix: str = "id") -> str:
    """Return an NCName-ish identifier stable enough for TEI ``xml:id``."""
    raw = str(value or "").strip()
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-.")
    if not raw or not re.match(r"[A-Za-z_]", raw[0]):
        raw = f"{prefix}-{raw}" if raw else prefix
    return raw


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _snapshot_digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _publication_projection(doc: UnifiedDocument) -> dict[str, object]:
    """Canonical TEI-visible projection used to detect stale embedded snapshots.

    The lossless snapshot is Novel Formatter-private state. A third-party TEI
    editor may legitimately alter the standard ``text``/``facsimile`` layer
    without touching ``xenoData``. Hashing this small semantic projection lets
    the importer detect that divergence instead of silently restoring stale
    publication text from an otherwise intact snapshot. Values are normalized
    exactly as the standard TEI projection is serialized (portable basenames,
    default facsimile dimensions), not as private workspace paths are stored.
    """
    metadata = {
        "title": str(getattr(doc.metadata, "title", "") or "Untitled"),
        "author": str(getattr(doc.metadata, "author", "") or ""),
        "publisher": str(getattr(doc.metadata, "publisher", "") or ""),
        "language": str(getattr(doc.metadata, "language", "") or "und"),
        "series": str(getattr(doc.metadata, "series", "") or ""),
        "volume": str(getattr(doc.metadata, "volume", "") or ""),
        "isbn": str(getattr(doc.metadata, "isbn", "") or ""),
        "description": str(getattr(doc.metadata, "description", "") or ""),
    }
    page_map = _page_lookup(doc)
    pages: list[dict[str, object]] = []
    for page_no in _all_page_numbers(doc):
        page = page_map.get(page_no)
        width = max(1, int(getattr(page, "width", 0) or 10000))
        height = max(1, int(getattr(page, "height", 0) or 10000))
        page_type = str(
            getattr(getattr(page, "page_type", None), "value", getattr(page, "page_type", "unknown"))
            or "unknown"
        )
        image_path = str(getattr(page, "image_path", "") or "")
        pages.append({
            "page_no": int(page_no),
            "page_type": page_type,
            "image_path": Path(image_path).name if image_path else "",
            "width": width,
            "height": height,
        })
    blocks: list[dict[str, object]] = []
    for index, block in enumerate(doc.blocks):
        bbox = getattr(block, "bbox", None)
        box = None
        if bbox is not None:
            box = [round(float(getattr(bbox, key, 0.0) or 0.0), 8) for key in ("x", "y", "w", "h")]
        image_path = str(getattr(block, "image_path", "") or "")
        blocks.append({
            # Compare the normalized public xml:id rather than the private raw
            # ID. This catches third-party edits to anchors/targets while still
            # allowing IDs that need TEI-safe prefixing to round-trip.
            "xml_id": f"b-{_safe_xml_id(block.id or str(index), 'block')}",
            "type": str(getattr(getattr(block, "type", None), "value", getattr(block, "type", "paragraph")) or "paragraph"),
            "text": str(getattr(block, "text", "") or ""),
            "page": int(_page_number_for_block(block) or 0),
            "bbox": box,
            "image_path": Path(image_path).name if image_path else "",
        })
    return {"metadata": metadata, "pages": pages, "blocks": blocks}


def _publication_projection_v1(doc: UnifiedDocument) -> dict[str, object]:
    """Phase39 public projection retained only for backwards verification."""
    page_map = _page_lookup(doc)
    pages: list[dict[str, object]] = []
    for page_no in _all_page_numbers(doc):
        page = page_map.get(page_no)
        width = max(1, int(getattr(page, "width", 0) or 10000))
        height = max(1, int(getattr(page, "height", 0) or 10000))
        page_type = str(
            getattr(getattr(page, "page_type", None), "value", getattr(page, "page_type", "unknown"))
            or "unknown"
        )
        image_path = str(getattr(page, "image_path", "") or "")
        pages.append({
            "page_no": int(page_no),
            "page_type": page_type,
            "image_path": Path(image_path).name if image_path else "",
            "width": width,
            "height": height,
        })
    blocks: list[dict[str, object]] = []
    for block in doc.blocks:
        bbox = getattr(block, "bbox", None)
        box = None
        if bbox is not None:
            box = [round(float(getattr(bbox, key, 0.0) or 0.0), 8) for key in ("x", "y", "w", "h")]
        image_path = str(getattr(block, "image_path", "") or "")
        blocks.append({
            "type": str(getattr(getattr(block, "type", None), "value", getattr(block, "type", "paragraph")) or "paragraph"),
            "text": str(getattr(block, "text", "") or ""),
            "page": int(_page_number_for_block(block) or 0),
            "bbox": box,
            "image_path": Path(image_path).name if image_path else "",
        })
    return {"pages": pages, "blocks": blocks}


def _publication_projection_digest(doc: UnifiedDocument) -> str:
    return _snapshot_digest(_json_text(_publication_projection(doc)))


def _publication_projection_digest_v1(doc: UnifiedDocument) -> str:
    return _snapshot_digest(_json_text(_publication_projection_v1(doc)))


def _provenance_projection_digest(doc: UnifiedDocument) -> str:
    items: list[dict[str, object]] = []
    for index, block in enumerate(doc.blocks):
        payload = _provenance_payload(block)
        if not payload:
            continue
        block_id = _safe_xml_id(block.id or str(index), "block")
        items.append({"target": f"#b-{block_id}", "payload": payload})
    items.sort(key=lambda item: str(item.get("target") or ""))
    return _snapshot_digest(_json_text(items))


def _tei_provenance_projection_digest(root: ET.Element) -> str:
    items: list[dict[str, object]] = []
    for ann in root.findall(f".//{_tei('standOff')}//{_tei('annotation')}"):
        if str(ann.attrib.get("type") or "") != "novel-formatter-provenance":
            continue
        target = str(ann.attrib.get("target") or "")
        note = ann.find(_tei("note"))
        if note is None:
            payload: object = {}
        else:
            try:
                payload = json.loads(str(note.text or "{}"))
            except Exception as exc:
                raise ValueError("TEI standOff OCR/裁决证据 JSON 无法解析") from exc
        items.append({"target": target, "payload": payload})
    items.sort(key=lambda item: str(item.get("target") or ""))
    return _snapshot_digest(_json_text(items))


def _page_number_for_block(block: Block) -> int:
    if block.page_number is not None:
        try:
            return int(block.page_number)
        except Exception:
            pass
    if block.page_index is not None:
        try:
            return int(block.page_index) + 1
        except Exception:
            pass
    try:
        return int(block.page or 0)
    except Exception:
        return 0


def _page_lookup(doc: UnifiedDocument) -> dict[int, PageInfo]:
    pages: dict[int, PageInfo] = {}
    for page in doc.pages:
        try:
            pages[int(page.page_no)] = page
        except Exception:
            continue
    return pages


def _all_page_numbers(doc: UnifiedDocument) -> list[int]:
    numbers = set(_page_lookup(doc))
    for block in doc.blocks:
        page_no = _page_number_for_block(block)
        if page_no > 0:
            numbers.add(page_no)
    return sorted(numbers)


def _zone_box(block: Block, width: int, height: int) -> tuple[int, int, int, int] | None:
    bbox = block.bbox
    if bbox is None:
        return None
    width = max(1, int(width or 10000))
    height = max(1, int(height or 10000))
    try:
        x = max(0.0, min(1.0, float(bbox.x)))
        y = max(0.0, min(1.0, float(bbox.y)))
        w = max(0.0, min(1.0 - x, float(bbox.w)))
        h = max(0.0, min(1.0 - y, float(bbox.h)))
    except Exception:
        return None
    ulx = int(round(x * width))
    uly = int(round(y * height))
    lrx = int(round((x + w) * width))
    lry = int(round((y + h) * height))
    return ulx, uly, max(ulx, lrx), max(uly, lry)


def _append_header(root: ET.Element, doc: UnifiedDocument, *, include_snapshot: bool) -> None:
    header = ET.SubElement(root, _tei("teiHeader"))
    file_desc = ET.SubElement(header, _tei("fileDesc"))
    title_stmt = ET.SubElement(file_desc, _tei("titleStmt"))
    ET.SubElement(title_stmt, _tei("title")).text = str(doc.metadata.title or "Untitled")
    if str(doc.metadata.author or "").strip():
        ET.SubElement(title_stmt, _tei("author")).text = str(doc.metadata.author)
    resp_stmt = ET.SubElement(title_stmt, _tei("respStmt"), {_xml_attr("id"): "novel-formatter"})
    ET.SubElement(resp_stmt, _tei("resp")).text = "OCR adjudication, structural editing, and TEI export"
    ET.SubElement(resp_stmt, _tei("name")).text = "Novel Formatter"

    publication_stmt = ET.SubElement(file_desc, _tei("publicationStmt"))
    if str(doc.metadata.publisher or "").strip():
        ET.SubElement(publication_stmt, _tei("publisher")).text = str(doc.metadata.publisher)
    else:
        ET.SubElement(publication_stmt, _tei("p")).text = "Unpublished digital transcription."
    if str(getattr(doc.metadata, "isbn", "") or "").strip():
        ET.SubElement(publication_stmt, _tei("idno"), {"type": "ISBN"}).text = str(doc.metadata.isbn)

    series = str(getattr(doc.metadata, "series", "") or "").strip()
    volume = str(getattr(doc.metadata, "volume", "") or "").strip()
    if series or volume:
        series_stmt = ET.SubElement(file_desc, _tei("seriesStmt"))
        if series:
            ET.SubElement(series_stmt, _tei("title"), {"level": "s"}).text = series
        if volume:
            ET.SubElement(series_stmt, _tei("biblScope"), {"unit": "volume"}).text = volume

    description = str(getattr(doc.metadata, "description", "") or "").strip()
    if description:
        notes_stmt = ET.SubElement(file_desc, _tei("notesStmt"))
        ET.SubElement(notes_stmt, _tei("note"), {"type": "description"}).text = description

    source_desc = ET.SubElement(file_desc, _tei("sourceDesc"))
    source_text = "Digital transcription derived from page images/PDF and OCR."
    if str(getattr(doc.metadata, "source_engine", "") or "").strip():
        source_text += f" OCR source: {doc.metadata.source_engine}."
    ET.SubElement(source_desc, _tei("p")).text = source_text

    encoding_desc = ET.SubElement(header, _tei("encodingDesc"))
    project_desc = ET.SubElement(encoding_desc, _tei("projectDesc"))
    ET.SubElement(project_desc, _tei("p")).text = (
        "TEI is an interchange layer. Publication text is separated from OCR/adjudication evidence; "
        "page geometry is linked through facsimile zones."
    )
    app_info = ET.SubElement(encoding_desc, _tei("appInfo"))
    application = ET.SubElement(
        app_info,
        _tei("application"),
        {"ident": "novel-formatter", "version": "2.0"},
    )
    ET.SubElement(application, _tei("label")).text = "Novel Formatter"

    profile_desc = ET.SubElement(header, _tei("profileDesc"))
    lang_usage = ET.SubElement(profile_desc, _tei("langUsage"))
    lang = str(getattr(doc.metadata, "language", "") or "und")
    ET.SubElement(lang_usage, _tei("language"), {"ident": lang}).text = lang

    if include_snapshot:
        xeno = ET.SubElement(header, _tei("xenoData"), {"type": "novel-formatter-snapshot"})
        snapshot_text = _json_text(doc.to_dict())
        snap = ET.SubElement(
            xeno,
            _nf("documentSnapshot"),
            {
                "mediaType": "application/json",
                "schema": SNAPSHOT_SCHEMA,
                "hashAlgorithm": SNAPSHOT_HASH,
                "sha256": _snapshot_digest(snapshot_text),
                "byteLength": str(len(snapshot_text.encode("utf-8"))),
                "projectionVersion": PROJECTION_VERSION,
                "projectionSha256": _publication_projection_digest(doc),
                "provenanceSha256": _provenance_projection_digest(doc),
            },
        )
        snap.text = snapshot_text


def _append_facsimile(root: ET.Element, doc: UnifiedDocument) -> dict[int, str]:
    page_map = _page_lookup(doc)
    page_numbers = _all_page_numbers(doc)
    if not page_numbers:
        return {}

    facsimile = ET.SubElement(root, _tei("facsimile"))
    block_zone_ids: dict[int, str] = {}
    blocks_by_page: dict[int, list[tuple[int, Block]]] = {}
    for index, block in enumerate(doc.blocks):
        page_no = _page_number_for_block(block)
        if page_no > 0:
            blocks_by_page.setdefault(page_no, []).append((index, block))

    for page_no in page_numbers:
        info = page_map.get(page_no)
        width = max(1, int(getattr(info, "width", 0) or 10000))
        height = max(1, int(getattr(info, "height", 0) or 10000))
        attrs = {
            _xml_attr("id"): f"surface-p{page_no:05d}",
            "n": str(page_no),
            "ulx": "0",
            "uly": "0",
            "lrx": str(width),
            "lry": str(height),
        }
        if info is not None:
            attrs["type"] = info.page_type.value
        surface = ET.SubElement(facsimile, _tei("surface"), attrs)
        if info is not None and str(info.image_path or "").strip():
            # Keep the TEI portable; exact local paths remain in xenoData snapshot.
            ET.SubElement(surface, _tei("graphic"), {"url": Path(info.image_path).name})

        for block_index, block in blocks_by_page.get(page_no, []):
            box = _zone_box(block, width, height)
            if box is None:
                continue
            block_id = _safe_xml_id(block.id or str(block_index), "block")
            zone_id = f"zone-{block_id}"
            ulx, uly, lrx, lry = box
            ET.SubElement(
                surface,
                _tei("zone"),
                {
                    _xml_attr("id"): zone_id,
                    "type": block.type.value,
                    "ulx": str(ulx),
                    "uly": str(uly),
                    "lrx": str(lrx),
                    "lry": str(lry),
                },
            )
            block_zone_ids[block_index] = zone_id
    return block_zone_ids


def _block_attrs(block: Block, index: int, zone_id: str | None = None) -> dict[str, str]:
    block_id = _safe_xml_id(block.id or str(index), "block")
    attrs = {_xml_attr("id"): f"b-{block_id}"}
    if zone_id:
        attrs["facs"] = f"#{zone_id}"
    page_no = _page_number_for_block(block)
    if page_no > 0:
        attrs["n"] = str(page_no)
    return attrs


def _append_text(root: ET.Element, doc: UnifiedDocument, zone_ids: dict[int, str]) -> None:
    lang = str(getattr(doc.metadata, "language", "") or "und")
    text = ET.SubElement(root, _tei("text"), {_xml_attr("lang"): lang})
    body = ET.SubElement(text, _tei("body"))
    current_container = body
    chapter_serial = 0

    for index, block in enumerate(doc.blocks):
        if (block.metadata or {}).get("consumed"):
            continue
        raw_text = str(block.text or "")
        if block.type != BlockType.IMAGE_REF and not raw_text.strip():
            continue
        attrs = _block_attrs(block, index, zone_ids.get(index))

        if block.type == BlockType.CHAPTER:
            chapter_serial += 1
            div_attrs = {
                "type": "chapter",
                "n": str(block.chapter_index or chapter_serial),
                _xml_attr("id"): f"chapter-{block.chapter_index or chapter_serial}",
            }
            current_container = ET.SubElement(body, _tei("div"), div_attrs)
            head = ET.SubElement(current_container, _tei("head"), attrs)
            head.text = raw_text
            continue

        if block.type == BlockType.SECTION:
            head = ET.SubElement(current_container, _tei("head"), {**attrs, "type": "section"})
            head.text = raw_text
        elif block.type == BlockType.FOOTNOTE:
            note = ET.SubElement(current_container, _tei("note"), {**attrs, "place": "foot"})
            note.text = raw_text
        elif block.type == BlockType.IMAGE_REF:
            figure = ET.SubElement(current_container, _tei("figure"), attrs)
            image_path = str(block.image_path or "").strip()
            if image_path:
                ET.SubElement(figure, _tei("graphic"), {"url": Path(image_path).name})
            desc = str((block.metadata or {}).get("alt_text") or "").strip()
            if desc:
                ET.SubElement(figure, _tei("figDesc")).text = desc
        elif block.type in {BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.RUBY}:
            p_attrs = dict(attrs)
            if block.type != BlockType.PARAGRAPH:
                p_attrs["type"] = block.type.value
            p = ET.SubElement(current_container, _tei("p"), p_attrs)
            p.text = raw_text
        else:
            ab = ET.SubElement(current_container, _tei("ab"), {**attrs, "type": block.type.value})
            ab.text = raw_text


def _provenance_payload(block: Block) -> dict:
    payload: dict[str, object] = {}
    if str(block.ocr_raw or ""):
        payload["ocr_raw"] = str(block.ocr_raw)
    if str(block.modified_by or ""):
        payload["modified_by"] = str(block.modified_by)
    if float(block.confidence or 0.0) != 1.0:
        payload["confidence"] = round(float(block.confidence), 6)
    if block.metadata:
        payload["metadata"] = block.metadata
    if block.source_format:
        payload["source_format"] = block.source_format
    if block.text_direction:
        payload["text_direction"] = block.text_direction
    if block.reading_order:
        payload["reading_order"] = block.reading_order
    if block.order_in_page is not None:
        payload["order_in_page"] = block.order_in_page
    return payload


def _append_standoff(root: ET.Element, doc: UnifiedDocument) -> None:
    annotated: list[tuple[int, Block, dict]] = []
    for index, block in enumerate(doc.blocks):
        payload = _provenance_payload(block)
        if payload:
            annotated.append((index, block, payload))
    if not annotated:
        return

    stand_off = ET.SubElement(root, _tei("standOff"), {"type": "novel-formatter-provenance"})
    items = ET.SubElement(stand_off, _tei("listAnnotation"), {"type": "ocr-adjudication"})
    for index, block, payload in annotated:
        block_id = _safe_xml_id(block.id or str(index), "block")
        ann = ET.SubElement(
            items,
            _tei("annotation"),
            {
                _xml_attr("id"): f"ann-{block_id}",
                "target": f"#b-{block_id}",
                "type": "novel-formatter-provenance",
            },
        )
        note = ET.SubElement(ann, _tei("note"), {"type": "application/json"})
        note.text = _json_text(payload)


def document_to_tei(doc: UnifiedDocument, *, include_snapshot: bool = True) -> str:
    """Serialize a :class:`UnifiedDocument` as TEI P5 XML."""
    lang = str(getattr(doc.metadata, "language", "") or "und")
    root = ET.Element(
        _tei("TEI"),
        {
            "version": TEI_VERSION,
            _xml_attr("lang"): lang,
        },
    )
    _append_header(root, doc, include_snapshot=include_snapshot)
    zone_ids = _append_facsimile(root, doc)
    _append_standoff(root, doc)
    _append_text(root, doc, zone_ids)
    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:  # Python 3.8 compatibility for downstream forks
        pass
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return xml_bytes.decode("utf-8") + "\n"


def export_tei(doc: UnifiedDocument, output_path: str | Path, *, include_snapshot: bool = True) -> str:
    target = Path(output_path)
    if target.suffix.lower() not in {".xml", ".tei"}:
        target = target.with_suffix(".xml")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = document_to_tei(doc, include_snapshot=include_snapshot)
    atomic_write_text(target, payload, encoding="utf-8")
    return str(target)


def _local(tag: str) -> str:
    return str(tag).split("}")[-1]


def _namespace(tag: str) -> str:
    text = str(tag)
    if text.startswith("{") and "}" in text:
        return text[1:].split("}", 1)[0]
    return ""


def _best_text(element: ET.Element) -> str:
    """Extract readable text from generic TEI without duplicating alternatives.

    ``choice`` prefers corr/reg/expan; TEI ruby emits the base (rb) only. This is
    only the external-TEI fallback path. Novel Formatter's own exports restore
    losslessly from xenoData before this function is used.
    """
    def walk(node: ET.Element) -> str:
        name = _local(node.tag)
        if name == "choice":
            children = list(node)
            priority = ("corr", "reg", "expan", "orig", "sic", "abbr")
            for wanted in priority:
                for child in children:
                    if _local(child.tag) == wanted:
                        return walk(child)
        if name == "ruby":
            for child in node:
                if _local(child.tag) == "rb":
                    return walk(child)
        pieces: list[str] = []
        if node.text:
            pieces.append(node.text)
        for child in node:
            child_name = _local(child.tag)
            if child_name == "rt":
                if child.tail:
                    pieces.append(child.tail)
                continue
            if child_name == "lb":
                pieces.append("\n")
            else:
                pieces.append(walk(child))
            if child.tail:
                pieces.append(child.tail)
        return "".join(pieces)
    return walk(element).strip()


def _parse_snapshot(root: ET.Element) -> tuple[UnifiedDocument | None, dict[str, object]]:
    """Return the embedded canonical snapshot plus an explicit integrity report.

    Phase38 exports did not carry a checksum, so checksum-less snapshots remain
    readable for compatibility.  Phase39+ exports are fail-closed: if the
    declared digest/length does not match, the importer raises instead of
    silently falling back to generic TEI and discarding OCR/adjudication state.
    """
    snap = root.find(f".//{{{NF_NS}}}documentSnapshot")
    if snap is None:
        return None, {"snapshot_present": False, "snapshot_verified": False}
    payload = str(snap.text or "")
    if not payload.strip():
        raise ValueError("TEI 内含 Novel Formatter 快照，但快照内容为空；为避免证据丢失，已拒绝降级导入")
    schema = str(snap.attrib.get("schema") or "")
    if schema not in {"", SNAPSHOT_SCHEMA}:
        raise ValueError(f"不支持的 Novel Formatter TEI 快照 schema：{schema}")

    expected_hash = str(snap.attrib.get("sha256") or "").strip().lower()
    algorithm = str(snap.attrib.get("hashAlgorithm") or "").strip().lower()
    expected_length = str(snap.attrib.get("byteLength") or "").strip()
    verified = False
    if expected_hash:
        if algorithm not in {"", SNAPSHOT_HASH}:
            raise ValueError(f"不支持的 TEI 快照哈希算法：{algorithm}")
        actual_hash = _snapshot_digest(payload)
        if actual_hash != expected_hash:
            raise ValueError("TEI 内嵌 Novel Formatter 快照 SHA-256 校验失败；文件可能被修改或损坏")
        verified = True
    if expected_length:
        try:
            declared_length = int(expected_length)
        except ValueError as exc:
            raise ValueError("TEI 快照 byteLength 非法") from exc
        actual_length = len(payload.encode("utf-8"))
        if declared_length != actual_length:
            raise ValueError("TEI 内嵌 Novel Formatter 快照长度校验失败；文件可能被修改或损坏")

    try:
        data = json.loads(payload)
    except Exception as exc:
        raise ValueError("TEI 内嵌 Novel Formatter 快照 JSON 无法解析；已拒绝有损降级") from exc
    if not isinstance(data, dict):
        raise ValueError("TEI 内嵌 Novel Formatter 快照不是 JSON 对象；已拒绝有损降级")
    doc = UnifiedDocument.from_dict(data)
    report = {
        "snapshot_present": True,
        "snapshot_verified": verified,
        "snapshot_schema": schema or SNAPSHOT_SCHEMA,
        "snapshot_sha256": expected_hash,
        "projection_sha256": str(snap.attrib.get("projectionSha256") or "").strip().lower(),
        "projection_version": str(snap.attrib.get("projectionVersion") or "").strip(),
        "provenance_sha256": str(snap.attrib.get("provenanceSha256") or "").strip().lower(),
        "tei_version": str(root.attrib.get("version") or ""),
    }
    setattr(doc, "tei_import_report", report)
    return doc, report


def _parse_page_type(value: str) -> BlockType:
    try:
        return BlockType(str(value))
    except Exception:
        return BlockType.UNKNOWN


def _parse_external_tei(root: ET.Element) -> UnifiedDocument:
    doc = UnifiedDocument()
    lang = str(root.attrib.get(_xml_attr("lang")) or "").strip()
    if lang:
        doc.metadata.language = lang

    title = root.find(f".//{_tei('teiHeader')}/{_tei('fileDesc')}/{_tei('titleStmt')}/{_tei('title')}")
    author = root.find(f".//{_tei('teiHeader')}/{_tei('fileDesc')}/{_tei('titleStmt')}/{_tei('author')}")
    publisher = root.find(f".//{_tei('teiHeader')}/{_tei('fileDesc')}/{_tei('publicationStmt')}/{_tei('publisher')}")
    isbn = root.find(f".//{_tei('teiHeader')}/{_tei('fileDesc')}/{_tei('publicationStmt')}/{_tei('idno')}[@type='ISBN']")
    series = root.find(f".//{_tei('teiHeader')}/{_tei('fileDesc')}/{_tei('seriesStmt')}/{_tei('title')}")
    volume = root.find(f".//{_tei('teiHeader')}/{_tei('fileDesc')}/{_tei('seriesStmt')}/{_tei('biblScope')}[@unit='volume']")
    if volume is None:
        volume = root.find(f".//{_tei('teiHeader')}/{_tei('fileDesc')}/{_tei('seriesStmt')}/{_tei('biblScope')}[@unit='vol']")
    description = root.find(f".//{_tei('teiHeader')}/{_tei('fileDesc')}/{_tei('notesStmt')}/{_tei('note')}[@type='description']")
    if title is not None:
        doc.metadata.title = _best_text(title)
    if author is not None:
        doc.metadata.author = _best_text(author)
    if publisher is not None:
        doc.metadata.publisher = _best_text(publisher)
    if isbn is not None:
        doc.metadata.isbn = _best_text(isbn)
    if series is not None:
        doc.metadata.series = _best_text(series)
    if volume is not None:
        doc.metadata.volume = _best_text(volume)
    if description is not None:
        doc.metadata.description = _best_text(description)
    doc.metadata.source_engine = "TEI P5 import"

    zone_map: dict[str, tuple[int, BoundingBox]] = {}
    surface_page_map: dict[str, int] = {}
    for surface in root.findall(f".//{_tei('facsimile')}//{_tei('surface')}"):
        try:
            page_no = int(surface.attrib.get("n") or len(doc.pages) + 1)
        except Exception:
            page_no = len(doc.pages) + 1
        try:
            sulx = float(surface.attrib.get("ulx") or 0)
            suly = float(surface.attrib.get("uly") or 0)
            slrx = float(surface.attrib.get("lrx") or 10000)
            slry = float(surface.attrib.get("lry") or 10000)
            width = max(1, int(round(slrx - sulx)))
            height = max(1, int(round(slry - suly)))
        except Exception:
            sulx = suly = 0.0
            width = height = 10000
        surface_id = str(surface.attrib.get(_xml_attr("id")) or "").strip()
        if surface_id:
            surface_page_map[surface_id] = page_no
        graphic = surface.find(_tei("graphic"))
        image_path = str(graphic.attrib.get("url") or "") if graphic is not None else ""
        doc.pages.append(PageInfo(
            page_no=page_no,
            page_type=_parse_page_type(surface.attrib.get("type") or "unknown"),
            image_path=image_path,
            width=width,
            height=height,
        ))
        for zone in surface.findall(_tei("zone")):
            zid = str(zone.attrib.get(_xml_attr("id")) or "").strip()
            if not zid:
                continue
            try:
                ulx = float(zone.attrib.get("ulx") or 0)
                uly = float(zone.attrib.get("uly") or 0)
                lrx = float(zone.attrib.get("lrx") or ulx)
                lry = float(zone.attrib.get("lry") or uly)
                bbox = BoundingBox(
                    x=max(0.0, (ulx - sulx) / width),
                    y=max(0.0, (uly - suly) / height),
                    w=max(0.0, (lrx - ulx) / width),
                    h=max(0.0, (lry - uly) / height),
                )
                zone_map[zid] = (page_no, bbox)
            except Exception:
                continue

    provenance: dict[str, dict] = {}
    for ann in root.findall(f".//{_tei('standOff')}//{_tei('annotation')}"):
        target = str(ann.attrib.get("target") or "").lstrip("#")
        note = ann.find(_tei("note"))
        if not target or note is None:
            continue
        try:
            data = json.loads(str(note.text or ""))
        except Exception:
            continue
        if isinstance(data, dict):
            provenance[target] = data

    body = root.find(f".//{_tei('text')}/{_tei('body')}")
    if body is None:
        return doc

    chapter_index = 0
    reading_order = 0
    current_page = 0

    def add_block(element: ET.Element, block_type: BlockType, text_value: str = "") -> None:
        nonlocal reading_order, chapter_index, current_page
        reading_order += 1
        xml_id = str(element.attrib.get(_xml_attr("id")) or "").strip()
        internal_id = xml_id[2:] if xml_id.startswith("b-") else (xml_id or f"tei-{reading_order}")
        facs = str(element.attrib.get("facs") or "").lstrip("#")
        page_no = 0
        bbox = None
        if facs in zone_map:
            page_no, bbox = zone_map[facs]
        elif str(element.attrib.get("n") or "").isdigit():
            page_no = int(element.attrib["n"])
        elif current_page > 0:
            page_no = current_page
        prov = provenance.get(xml_id, {})
        block = Block(
            type=block_type,
            text=text_value,
            page=page_no,
            page_number=page_no if page_no > 0 else None,
            page_index=(page_no - 1) if page_no > 0 else None,
            bbox=bbox,
            reading_order=reading_order,
            confidence=float(prov.get("confidence", 1.0) or 1.0),
            ocr_raw=str(prov.get("ocr_raw") or ""),
            modified_by=str(prov.get("modified_by") or ""),
            metadata=prov.get("metadata") if isinstance(prov.get("metadata"), dict) else {},
            source_format=str(prov.get("source_format") or "tei"),
            text_direction=str(prov.get("text_direction") or "") or None,
            id=internal_id,
            chapter_index=chapter_index,
        )
        doc.blocks.append(block)

    def walk(container: ET.Element) -> None:
        nonlocal chapter_index, current_page
        for child in list(container):
            name = _local(child.tag)
            if name == "pb":
                facs = str(child.attrib.get("facs") or "").lstrip("#")
                raw_n = str(child.attrib.get("n") or "").strip()
                if facs and facs in surface_page_map:
                    current_page = surface_page_map[facs]
                elif raw_n.isdigit():
                    current_page = int(raw_n)
                elif current_page > 0:
                    current_page += 1
                else:
                    current_page = 1
            elif name == "div":
                is_chapter = str(child.attrib.get("type") or "").lower() in {"chapter", "textpart"}
                if is_chapter:
                    chapter_index += 1
                walk(child)
            elif name == "head":
                typ = BlockType.CHAPTER if str(container.attrib.get("type") or "").lower() in {"chapter", "textpart"} else BlockType.SECTION
                before = len(doc.blocks)
                add_block(child, typ, _best_text(child))
                if typ == BlockType.CHAPTER and len(doc.blocks) > before:
                    block_idx = len(doc.blocks) - 1
                    doc.blocks[-1].chapter_index = chapter_index or 1
                    doc.toc.append(TocEntry(_best_text(child), chapter_index or 1, block_idx))
            elif name == "p":
                ptype = str(child.attrib.get("type") or "").lower()
                typ = BlockType.DIALOGUE if ptype == "dialogue" else (BlockType.RUBY if ptype == "ruby" else BlockType.PARAGRAPH)
                add_block(child, typ, _best_text(child))
            elif name == "note":
                add_block(child, BlockType.FOOTNOTE, _best_text(child))
            elif name == "figure":
                graphic = child.find(_tei("graphic"))
                fig_desc = child.find(_tei("figDesc"))
                add_block(child, BlockType.IMAGE_REF, "")
                if graphic is not None and doc.blocks:
                    doc.blocks[-1].image_path = str(graphic.attrib.get("url") or "")
                if fig_desc is not None and doc.blocks:
                    alt_text = _best_text(fig_desc)
                    if alt_text:
                        doc.blocks[-1].metadata["alt_text"] = alt_text
            elif name == "ab":
                try:
                    typ = BlockType(str(child.attrib.get("type") or "paragraph"))
                except Exception:
                    typ = BlockType.PARAGRAPH
                add_block(child, typ, _best_text(child))
            else:
                walk(child)

    walk(body)
    return doc


def _generic_import_report(root: ET.Element) -> dict[str, object]:
    """Describe approximations made by the lightweight generic TEI importer.

    Novel Formatter's own TEI round-trips through the private snapshot and is
    lossless. Generic TEI is intentionally imported conservatively; this report
    makes any richer TEI constructs visible to the caller instead of silently
    implying that the full source semantics were preserved.
    """
    counts: dict[str, int] = {}
    for element in root.iter():
        local = _local(element.tag)
        counts[local] = counts.get(local, 0) + 1
    approximated = {
        name: counts[name]
        for name in (
            "app", "lem", "rdg", "gap", "unclear", "supplied", "del", "add",
            "table", "row", "cell", "list", "item", "milestone", "seg",
        )
        if counts.get(name)
    }
    warnings: list[str] = []
    if approximated:
        warnings.append(
            "外部 TEI 含 Novel Formatter 通用导入器未完整建模的结构："
            + ", ".join(f"{name}×{count}" for name, count in sorted(approximated.items()))
        )
    return {
        "snapshot_present": False,
        "snapshot_verified": False,
        "import_mode": "generic-tei",
        "tei_version": str(root.attrib.get("version") or ""),
        "pb_count": counts.get("pb", 0),
        "surface_count": counts.get("surface", 0),
        "zone_count": counts.get("zone", 0),
        "approximated_elements": approximated,
        "warnings": warnings,
    }


def load_tei(source: str | Path) -> UnifiedDocument:
    """Load Novel Formatter TEI losslessly, or import a generic TEI P5 file."""
    path = Path(source)
    tree = ET.parse(path)
    root = tree.getroot()
    if _local(root.tag) != "TEI":
        raise ValueError("不是 TEI P5 文档：根元素必须是 <TEI>")
    if _namespace(root.tag) != TEI_NS:
        raise ValueError(
            "不是规范 TEI P5 文档：<TEI> 根元素缺少或使用了错误的 TEI namespace"
        )
    snap_doc, snapshot_report = _parse_snapshot(root)
    if snap_doc is not None:
        expected_projection = str(snapshot_report.get("projection_sha256") or "").strip().lower()
        if expected_projection:
            projected = _parse_external_tei(root)
            projection_version = str(snapshot_report.get("projection_version") or "").strip()
            if projection_version in {"", "1"}:
                # Phase39 files had integrity hashes before public metadata/ID
                # coverage was added. Keep them verifiable instead of turning
                # a hardening upgrade into a breaking interchange change.
                actual_projection = _publication_projection_digest_v1(projected)
            elif projection_version == PROJECTION_VERSION:
                actual_projection = _publication_projection_digest(projected)
            else:
                raise ValueError(f"不支持的 TEI projectionVersion：{projection_version}")
            if actual_projection != expected_projection:
                raise ValueError(
                    "TEI 标准正文/facsimile/公共元数据 与内嵌 Novel Formatter 快照不一致；"
                    "检测到外部编辑或文件损坏，已拒绝静默恢复旧正文"
                )
            snapshot_report["projection_verified"] = True
        expected_provenance = str(snapshot_report.get("provenance_sha256") or "").strip().lower()
        if expected_provenance:
            actual_provenance = _tei_provenance_projection_digest(root)
            if actual_provenance != expected_provenance:
                raise ValueError(
                    "TEI standOff OCR/裁决证据与内嵌 Novel Formatter 快照不一致；"
                    "检测到外部编辑或文件损坏，已拒绝静默恢复旧证据"
                )
            snapshot_report["provenance_verified"] = True
        setattr(snap_doc, "tei_import_report", snapshot_report)
        return snap_doc
    doc = _parse_external_tei(root)
    setattr(doc, "tei_import_report", _generic_import_report(root))
    return doc


__all__ = [
    "TEI_NS",
    "NF_NS",
    "TEI_VERSION",
    "document_to_tei",
    "export_tei",
    "load_tei",
]
