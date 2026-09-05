#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Locked Ruby side-channel for the compact fusion+skeleton AI exchange bundle.

The external model edits plain authoritative text only.  Ruby readings are frozen
in a separate checksummed sidecar and are re-attached by a deterministic builder.
This keeps findtextCenterNet output independent from OCR voting and prevents an
external model from silently changing or losing readings while rebuilding EPUB.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from models.document import UnifiedDocument
from engine.ruby_anchor_core import ANCHOR_POLICY

LOCK_SCHEMA = "novel_formatter.ruby_exchange_lock.v1"
EDITS_SCHEMA = "novel_formatter.fusion_skeleton_edits.v1"


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=not pretty,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _annotation_pair_count(rows: list[dict]) -> int:
    return sum(len(row.get("annotations") or []) for row in rows)


def build_locked_ruby_payload(primary_doc: UnifiedDocument, fusion_package: dict) -> dict:
    """Freeze row-addressable Ruby annotations without making them editable text."""
    from engine.ai_repair_epub import build_repair_document

    repair_doc = build_repair_document(primary_doc, fusion_package, export_revision=1)
    rows: list[dict] = []
    for block in repair_doc.blocks:
        metadata = getattr(block, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        row_id = str(metadata.get("ai_repair_item_id", "") or "").strip()
        if not row_id:
            continue
        annotations = [
            copy.deepcopy(item)
            for item in (metadata.get("ruby_annotations") or [])
            if isinstance(item, dict)
            and str(item.get("base", "") or "")
            and str(item.get("reading", "") or "")
        ]
        rows.append({
            "row_id": row_id,
            "html_id": str(metadata.get("ai_repair_html_id", "") or ""),
            "page": int(getattr(block, "page", 0) or 0),
            "baseline_text": str(getattr(block, "text", "") or ""),
            "baseline_sha256": _sha256_text(str(getattr(block, "text", "") or "")),
            "annotations": annotations,
            "pair_count": len(annotations),
        })

    overlay = fusion_package.get("ruby_overlay") if isinstance(fusion_package, dict) else None
    overlay_meta = overlay.get("document_metadata") if isinstance(overlay, dict) else {}
    enabled = bool(
        isinstance(overlay, dict)
        and overlay_meta.get("ruby_preservation_enabled")
        and overlay.get("blocks")
    )
    payload = {
        "schema": LOCK_SCHEMA,
        "mode": "locked_side_channel",
        "ruby_preservation_enabled": enabled,
        "source_engine": str((overlay_meta or {}).get("ruby_preservation_engine", "") or ""),
        "rules": {
            "editable": False,
            "may_change_reading": False,
            "may_add_reading": False,
            "may_delete_reading": False,
            "reattach_policy": "exact_then_findtext_verified_edit_span",
            "base_correction_policy": ANCHOR_POLICY,
            "ambiguous_policy": "drop_ruby_never_guess",
            "authoritative_prose_source": "AI_OUTPUT/edited_text.json",
        },
        "rows": rows,
        "row_count": len(rows),
        "ruby_pair_count": _annotation_pair_count(rows),
        "anchor_policy_version": 4,
        "anchor_policy": ANCHOR_POLICY,
    }
    payload["payload_sha256"] = _sha256_bytes(
        _json_bytes({key: value for key, value in payload.items() if key != "payload_sha256"})
    )
    return payload


def build_edit_template(fusion_package: dict, *, fusion_sha256: str, ruby_lock_sha256: str) -> dict:
    rows = []
    for item in sorted(
        [value for value in (fusion_package.get("editable_items") or []) if isinstance(value, dict)],
        key=lambda value: int(value.get("row_index", 0) or 0),
    ):
        row_id = str(item.get("row_id") or item.get("item_id") or "").strip()
        if not row_id:
            continue
        text = str(item.get("edited_text", item.get("original_fused_text", "")) or "")
        rows.append({
            "row_id": row_id,
            "baseline_sha256": _sha256_text(text),
            "edited_text": text,
        })
    return {
        "schema": EDITS_SCHEMA,
        "package_id": str(fusion_package.get("package_id", "") or ""),
        "fusion_json_sha256": str(fusion_sha256),
        "ruby_lock_sha256": str(ruby_lock_sha256),
        "instructions": {
            "edit_only": "rows[].edited_text",
            "do_not_edit": [
                "row_id", "baseline_sha256", "fusion_json_sha256",
                "ruby_lock_sha256", "04_ruby_overlay.locked.json",
                "framework/structure_skeleton.epub",
            ],
            "ruby_policy": "Do not type furigana into edited_text. The builder reattaches locked Ruby automatically.",
        },
        "rows": rows,
    }


def model_command_text() -> str:
    return """# Novel Formatter — Fusion + Skeleton AI contract\n\nYou are editing authoritative **plain Japanese prose only**. Ruby/furigana is a locked side-channel.\n\n## Required workflow\n\n1. Read `01_multi_ocr_fusion_result.json` for OCR evidence/context.\n2. Edit only `AI_OUTPUT/edited_text.json` -> `rows[].edited_text`.\n3. Never edit `04_ruby_overlay.locked.json` or `framework/structure_skeleton.epub`.\n4. Do not insert Aozora Ruby markers, parentheses readings, `<ruby>`, or `<rt>` into edited text.\n5. Run `python3 tools/build_final_epub.py` from the package root.\n6. Run `python3 tools/validate_ruby.py AI_OUTPUT/final.epub`.\n7. Deliver `AI_OUTPUT/final.epub` plus `AI_OUTPUT/ruby_validation.json`.\n\n## Ruby safety\n\n- Existing readings are immutable evidence from findtextCenterNet.\n- If the edited prose still contains a uniquely identifiable Ruby base, the builder re-attaches it.\n- If ordinary OCR misspelled the Ruby base but locked findtext evidence recorded the correct base+reading, a small mapped substitution/insertion/deletion may migrate the reading to that exact findtext-observed base.\n- A semantic replacement that does not equal the locked findtext base is never treated as an OCR correction.\n- Repeated identical bases are tracked by their locked source span. Deleting one occurrence never transfers its reading to a surviving homograph.\n- If AI duplicates a Ruby-bearing phrase, the original occurrence must remain uniquely identifiable by the edit map; otherwise the Ruby is dropped.\n- If the base was deleted, becomes ambiguous, or the edit cannot be proven safe, that Ruby is dropped.\n- Never guess a Ruby location and never restore stale prose just to keep a reading.\n\nThe final EPUB must therefore be generated by the provided builder, not by rewriting the skeleton from scratch.\n"""


_BUILD_TOOL = r'''#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, hashlib, html, json, os, re, shutil, tempfile, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
from ruby_anchor import resolve_annotation_for_text

XHTML_NS = "http://www.w3.org/1999/xhtml"
ET.register_namespace("", XHTML_NS)
LOCK_SCHEMA = "novel_formatter.ruby_exchange_lock.v1"
EDITS_SCHEMA = "novel_formatter.fusion_skeleton_edits.v1"


def sha256_bytes(data): return hashlib.sha256(data).hexdigest()
def sha256_file(path): return sha256_bytes(Path(path).read_bytes())
def text_sha(value): return sha256_bytes(str(value or "").encode("utf-8"))

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def validate_plain_edit(row_id,text):
    value=str(text or "")
    if "\x00" in value: raise SystemExit("NUL is not allowed in edited text: %s"%row_id)
    if re.search(r"<\s*/?\s*(?:ruby|rt)\b",value,re.I): raise SystemExit("Ruby HTML is forbidden in edited text: %s"%row_id)
    if re.search(r"｜[^《\n]{1,128}《[^》\n]{1,128}》",value): raise SystemExit("Aozora Ruby markers are forbidden in edited text: %s"%row_id)
    return value

def placements(text, annotations, source_text=""):
    used=[]; out=[]; dropped=[]; migrated=[]
    for ann in annotations or []:
        if not isinstance(ann,dict): continue
        base=str(ann.get("base","") or ""); reading=str(ann.get("reading","") or "")
        if not base or not reading: continue
        resolved=resolve_annotation_for_text(text,ann,used,source_text=str(source_text or ""))
        if not resolved:
            dropped.append({"base":base,"reading":reading,"reason":"missing_ambiguous_or_unsafe_base_edit"}); continue
        migrated_ann=resolved.get("annotation") or ann
        base_now=str(migrated_ann.get("base","") or "")
        p=int(resolved.get("position",-1))
        if p<0 or not base_now:
            dropped.append({"base":base,"reading":reading,"reason":"invalid_resolved_anchor"}); continue
        end=p+len(base_now); used.append((p,end)); out.append((p,end,base_now,reading))
        if str(resolved.get("mode", ""))=="base_edit_span":
            migrated.append({"old_base":base,"new_base":base_now,"reading":reading,"position":p,"policy":migrated_ann.get("base_migration_policy","")})
    out.sort(key=lambda x:x[0])
    return out,dropped,migrated

def append_plain(parent,last,text):
    parts=str(text).split("\n")
    for idx,part in enumerate(parts):
        if part:
            if last is None: parent.text=(parent.text or "")+part
            else: last.tail=(last.tail or "")+part
        if idx < len(parts)-1:
            br=ET.SubElement(parent,"{%s}br"%XHTML_NS); last=br
    return last

def set_ruby_content(element,text,annotations,source_text=""):
    for child in list(element): element.remove(child)
    element.text=None
    last=None; cursor=0
    places,dropped,migrated=placements(text,annotations,source_text)
    for start,end,base,reading in places:
        last=append_plain(element,last,text[cursor:start])
        ruby=ET.SubElement(element,"{%s}ruby"%XHTML_NS); ruby.text=base
        rt=ET.SubElement(ruby,"{%s}rt"%XHTML_NS); rt.text=reading
        last=ruby; cursor=end
    append_plain(element,last,text[cursor:])
    return len(places),dropped,migrated

def local(tag): return tag.rsplit("}",1)[-1] if "}" in tag else tag

def verify_inputs(root,manifest,lock,edits):
    if lock.get("schema")!=LOCK_SCHEMA: raise SystemExit("invalid locked Ruby schema")
    if edits.get("schema")!=EDITS_SCHEMA: raise SystemExit("invalid edits schema")
    fusion=root/manifest["fusion_json"]; skeleton=root/manifest["skeleton_epub"]; lock_path=root/manifest["ruby_lock"]
    checks=((fusion,"fusion_json_sha256"),(skeleton,"skeleton_epub_sha256"),(lock_path,"ruby_lock_sha256"))
    for path,key in checks:
        got=sha256_file(path); expected=str(manifest.get(key,"") or "")
        if not expected or got!=expected: raise SystemExit("hash mismatch: %s"%path)
    if str(edits.get("fusion_json_sha256","") or "")!=manifest["fusion_json_sha256"]: raise SystemExit("edits bound to different fusion JSON")
    if str(edits.get("ruby_lock_sha256","") or "")!=manifest["ruby_lock_sha256"]: raise SystemExit("edits bound to different Ruby lock")
    for relative,expected in (manifest.get("tool_sha256") or {}).items():
        tool_path=root/str(relative)
        if not tool_path.is_file() or sha256_file(tool_path)!=str(expected or ""):
            raise SystemExit("tool hash mismatch: %s"%relative)
    payload_hash=sha256_bytes(json.dumps({k:v for k,v in lock.items() if k!="payload_sha256"},ensure_ascii=False,sort_keys=True,separators=(",",":")).encode())
    if payload_hash != str(lock.get("payload_sha256","") or ""): raise SystemExit("locked Ruby payload self-hash mismatch")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default=str(Path(__file__).resolve().parents[1])); ap.add_argument("--edits",default="AI_OUTPUT/edited_text.json"); ap.add_argument("--output",default="AI_OUTPUT/final.epub")
    args=ap.parse_args(); root=Path(args.root).resolve()
    manifest=load_json(root/"00_manifest.json"); lock=load_json(root/manifest["ruby_lock"]); edits=load_json(root/args.edits)
    verify_inputs(root,manifest,lock,edits)
    fusion=load_json(root/manifest["fusion_json"])
    baseline={str(i.get("row_id") or i.get("item_id") or ""):str(i.get("edited_text",i.get("original_fused_text","")) or "") for i in fusion.get("editable_items",[]) if isinstance(i,dict)}
    updates={str(r.get("row_id","") or ""):r for r in edits.get("rows",[]) if isinstance(r,dict)}
    if set(updates)!=set(baseline): raise SystemExit("edited row-id set differs from frozen fusion row-id set")
    for row_id,row in updates.items():
        if str(row.get("baseline_sha256","") or "") != text_sha(baseline[row_id]): raise SystemExit("baseline hash mismatch for %s"%row_id)
        row["edited_text"]=validate_plain_edit(row_id,row.get("edited_text", ""))
    lock_rows={str(r.get("row_id","") or ""):r for r in lock.get("rows",[]) if isinstance(r,dict)}
    skeleton=root/manifest["skeleton_epub"]; output=root/args.output; output.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix="nf_ruby_build_"))
    report={"schema":"novel_formatter.ruby_exchange_build_report.v2","rendered_pairs":0,"migrated_pairs":0,"migrations":[],"dropped_pairs":[],"rows":[]}
    try:
        with zipfile.ZipFile(skeleton,"r") as z: z.extractall(temp)
        found={}
        for path in temp.rglob("*.xhtml"):
            tree=ET.parse(path); root_el=tree.getroot(); changed=False
            for el in root_el.iter():
                rid=str(el.attrib.get("data-item-id","") or "")
                if rid and rid in updates:
                    if rid in found: raise SystemExit("duplicate data-item-id: %s"%rid)
                    found[rid]=(path,el); text=str(updates[rid].get("edited_text","") or ""); rowlock=lock_rows.get(rid,{})
                    rendered,dropped,migrated=set_ruby_content(el,text,rowlock.get("annotations") or [],rowlock.get("baseline_text", ""))
                    report["rendered_pairs"]+=rendered
                    report["migrated_pairs"]+=len(migrated)
                    report["migrations"].extend([{"row_id":rid,**m} for m in migrated])
                    report["dropped_pairs"].extend([{"row_id":rid,**d} for d in dropped])
                    report["rows"].append({"row_id":rid,"rendered_pairs":rendered,"migrated_pairs":len(migrated),"dropped_pairs":len(dropped),"edited_text_sha256":text_sha(text)})
                    for key in list(el.attrib):
                        if key.startswith("data-"): el.attrib.pop(key,None)
                    changed=True
            if changed: tree.write(path,encoding="utf-8",xml_declaration=True)
        missing=sorted(set(updates)-set(found))
        if missing: raise SystemExit("skeleton missing row ids: %s"%", ".join(missing[:8]))
        with zipfile.ZipFile(output,"w") as z:
            mime=temp/"mimetype"
            z.write(mime,"mimetype",compress_type=zipfile.ZIP_STORED)
            for path in sorted(temp.rglob("*")):
                if path.is_file() and path!=mime: z.write(path,path.relative_to(temp).as_posix(),compress_type=zipfile.ZIP_DEFLATED)
        report["output"]=str(output); report["output_sha256"]=sha256_file(output); report["ruby_pair_count_locked"]=int(lock.get("ruby_pair_count",0) or 0)
        (root/"AI_OUTPUT/ruby_build_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally: shutil.rmtree(temp,ignore_errors=True)
if __name__=="__main__": main()
'''

_VALIDATE_TOOL = r'''#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
from ruby_anchor import resolve_annotation_for_text

XHTML_NS="http://www.w3.org/1999/xhtml"
def local(tag): return tag.rsplit("}",1)[-1] if "}" in tag else tag
def load(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def expected_pairs(text, annotations, source_text=""):
    used=[]; out=[]; dropped=[]; migrated=[]
    for ann in annotations or []:
        if not isinstance(ann,dict): continue
        base=str(ann.get("base","") or ""); reading=str(ann.get("reading","") or "")
        if not base or not reading: continue
        resolved=resolve_annotation_for_text(text,ann,used,source_text=str(source_text or ""))
        if not resolved:
            dropped.append((base,reading)); continue
        migrated_ann=resolved.get("annotation") or ann
        base_now=str(migrated_ann.get("base","") or "")
        p=int(resolved.get("position",-1))
        if p<0 or not base_now:
            dropped.append((base,reading)); continue
        used.append((p,p+len(base_now))); out.append((p,base_now,reading))
        if str(resolved.get("mode", ""))=="base_edit_span": migrated.append((base,base_now,reading))
    return out,dropped,migrated

def plain_without_rt(el):
    parts=[]
    def walk(node, include_tail=True):
        if local(node.tag)=="rt":
            if include_tail and node.tail: parts.append(node.tail)
            return
        if node.text: parts.append(node.text)
        for c in list(node): walk(c, True)
        if include_tail and node.tail: parts.append(node.tail)
    walk(el, False); return "".join(parts).replace("\r\n","\n")

def ruby_pairs(el):
    out=[]
    for ruby in [c for c in el.iter() if local(c.tag)=="ruby"]:
        rt=next((c for c in list(ruby) if local(c.tag)=="rt"),None)
        out.append((str(ruby.text or ""),str(rt.text or "") if rt is not None else ""))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("epub",nargs="?",default="AI_OUTPUT/final.epub"); ap.add_argument("--root",default=str(Path(__file__).resolve().parents[1])); args=ap.parse_args()
    root=Path(args.root).resolve(); epub=(root/args.epub).resolve() if not Path(args.epub).is_absolute() else Path(args.epub)
    manifest=load(root/"00_manifest.json"); edits=load(root/"AI_OUTPUT/edited_text.json"); lock=load(root/manifest["ruby_lock"])
    for relative,expected_hash in (manifest.get("tool_sha256") or {}).items():
        tool_path=root/str(relative)
        if not tool_path.is_file() or sha(tool_path)!=str(expected_hash or ""):
            raise SystemExit("tool hash mismatch: %s"%relative)
    expected={str(r.get("row_id","") or ""):str(r.get("edited_text","") or "") for r in edits.get("rows",[]) if isinstance(r,dict)}
    lock_by_html={str(r.get("html_id","") or ""):r for r in lock.get("rows",[]) if isinstance(r,dict) and str(r.get("html_id","") or "")}
    result={"schema":"novel_formatter.ruby_exchange_validation.v3","epub":str(epub),"epub_sha256":sha(epub),"errors":[],"ruby_count":0,"rt_count":0,"checked_rows":0,"locked_pair_count":int(lock.get("ruby_pair_count",0) or 0),"expected_rendered_pairs":0,"expected_migrated_pairs":0,"expected_dropped_pairs":0}
    with zipfile.ZipFile(epub,"r") as z:
        infos=z.infolist()
        if not infos or infos[0].filename!="mimetype" or infos[0].compress_type!=zipfile.ZIP_STORED: result["errors"].append("invalid_mimetype_entry")
        if z.read("mimetype")!=b"application/epub+zip": result["errors"].append("invalid_mimetype_content")
        seen=set()
        for name in z.namelist():
            if not name.endswith(".xhtml"): continue
            root_el=ET.fromstring(z.read(name))
            for el in root_el.iter():
                if local(el.tag)=="ruby": result["ruby_count"]+=1
                if local(el.tag)=="rt": result["rt_count"]+=1
                if "data-item-id" in el.attrib: result["errors"].append("work_attribute_survived:%s"%name)
                html_id=str(el.attrib.get("id","") or "")
                rowlock=lock_by_html.get(html_id)
                if rowlock:
                    rid=str(rowlock.get("row_id","") or ""); seen.add(rid); result["checked_rows"]+=1
                    got=plain_without_rt(el); want=expected.get(rid,"")
                    if got!=want: result["errors"].append("plain_text_mismatch:%s"%rid)
                    safe,dropped,migrated=expected_pairs(want,rowlock.get("annotations") or [],rowlock.get("baseline_text", ""))
                    expected_row_pairs=[(base,reading) for _pos,base,reading in sorted(safe)]
                    actual_row_pairs=ruby_pairs(el)
                    result["expected_rendered_pairs"]+=len(expected_row_pairs); result["expected_migrated_pairs"]+=len(migrated); result["expected_dropped_pairs"]+=len(dropped)
                    if actual_row_pairs!=expected_row_pairs:
                        result["errors"].append("ruby_pair_mismatch:%s"%rid)
        missing=sorted(set(expected)-seen)
        if missing: result["errors"].append("missing_rows:"+",".join(missing[:8]))
    if result["ruby_count"]!=result["rt_count"]: result["errors"].append("ruby_rt_count_mismatch")
    if result["ruby_count"]!=result["expected_rendered_pairs"]: result["errors"].append("rendered_ruby_count_mismatch")
    result["ok"]=not result["errors"]
    out=root/"AI_OUTPUT/ruby_validation.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result["ok"] else 2)
if __name__=="__main__": main()
'''


def write_exchange_tools(folder: Path) -> list[str]:
    tools = folder / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    anchor_source = (Path(__file__).with_name("ruby_anchor_core.py")).read_text(encoding="utf-8")
    files = {
        "tools/ruby_anchor.py": anchor_source,
        "tools/build_final_epub.py": _BUILD_TOOL,
        "tools/validate_ruby.py": _VALIDATE_TOOL,
    }
    for relative, content in files.items():
        path = folder / relative
        path.write_text(content, encoding="utf-8")
        try:
            path.chmod(0o755)
        except OSError:
            pass
    return list(files)


__all__ = [
    "LOCK_SCHEMA", "EDITS_SCHEMA", "build_locked_ruby_payload",
    "build_edit_template", "model_command_text", "write_exchange_tools",
]
