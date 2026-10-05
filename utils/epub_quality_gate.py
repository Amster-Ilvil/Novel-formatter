# -*- coding: utf-8 -*-
"""Always-on deterministic EPUB structural quality gate.

The project targets the stable W3C EPUB 3.3 Recommendation for publication,
while keeping the checks compatible with the EPUB 3.4 Candidate Recommendation.
This module deliberately covers high-value invariants locally so a missing Java
runtime never turns a broken package into a successful export.  Official
EPUBCheck remains the final external conformance checker when available.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
import posixpath
import re
import struct
import unicodedata
import zipfile
import xml.etree.ElementTree as ET

from utils.safe_archive import validate_zip

CONTAINER_PATH = "META-INF/container.xml"
MIMETYPE = "application/epub+zip"
OCF_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
OPF_NS = "http://www.idpf.org/2007/opf"
DC_NS = "http://purl.org/dc/elements/1.1/"
XHTML_NS = "http://www.w3.org/1999/xhtml"
EPUB_NS = "http://www.idpf.org/2007/ops"
XML_NS = "http://www.w3.org/XML/1998/namespace"
STABLE_TARGET = "EPUB 3.3"
DRAFT_COMPAT = "EPUB 3.4 CR"
_MODIFIED_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")

_IMAGE_SIGNATURES = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/gif": (b"GIF87a", b"GIF89a"),
}


def _zip_local_header(path: Path, info: zipfile.ZipInfo) -> dict[str, int | str]:
    """Read the physical local-file header, not only the central directory.

    Python's ``ZipInfo.extra`` is sourced from the central directory. OCF's
    special ``mimetype`` rule applies to its ZIP header, so check the local
    record directly as well. This also detects prefixed bytes before the first
    ZIP entry and local/central metadata disagreement.
    """
    with path.open("rb") as fh:
        fh.seek(int(info.header_offset))
        header = fh.read(30)
        if len(header) != 30:
            raise ValueError("ZIP local header 截断")
        (
            signature, extract_version, flag_bits, compression,
            _mtime, _mdate, _crc32, _compressed_size, _file_size,
            filename_len, extra_len,
        ) = struct.unpack("<IHHHHHIIIHH", header)
        if signature != 0x04034B50:
            raise ValueError("ZIP local header 签名错误")
        raw_name = fh.read(filename_len)
        try:
            filename = raw_name.decode("utf-8" if flag_bits & 0x800 else "cp437")
        except Exception:
            filename = raw_name.decode("utf-8", errors="replace")
        return {
            "header_offset": int(info.header_offset),
            "extract_version": int(extract_version),
            "flag_bits": int(flag_bits),
            "compression": int(compression),
            "filename_len": int(filename_len),
            "extra_len": int(extra_len),
            "filename": filename,
        }


@dataclass
class EpubQualityGateReport:
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, object] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        self.valid = False
        self.errors.append(str(message))

    def add_warning(self, message: str) -> None:
        self.warnings.append(str(message))

    def summary(self) -> str:
        if self.valid and not self.warnings:
            return "内置 EPUB 质量闸门通过 ✓"
        if self.valid:
            return f"内置 EPUB 质量闸门通过，{len(self.warnings)} 条警告"
        return f"内置 EPUB 质量闸门失败：{len(self.errors)} 个错误、{len(self.warnings)} 条警告"

    def to_dict(self) -> dict:
        return asdict(self)


def _tag_local(tag: str) -> str:
    return str(tag).split("}")[-1]


def _namespace(tag: str) -> str:
    text = str(tag)
    if text.startswith("{") and "}" in text:
        return text[1:].split("}", 1)[0]
    return ""


def _resolve_href(opf_path: str, href: str) -> str:
    base = str(PurePosixPath(opf_path).parent)
    joined = posixpath.normpath(posixpath.join(base, href.split("#", 1)[0]))
    return joined.lstrip("./")


def _resolve_document_ref(document_path: str, href: str) -> tuple[str, str]:
    raw = str(href or "").strip()
    if "#" in raw:
        path_part, fragment = raw.split("#", 1)
    else:
        path_part, fragment = raw, ""
    if not path_part:
        return str(document_path), fragment
    base = str(PurePosixPath(document_path).parent)
    resolved = posixpath.normpath(posixpath.join(base, path_part)).lstrip("./")
    return resolved, fragment


def _is_external_ref(value: str) -> bool:
    text = str(value or "").strip().lower()
    return not text or text.startswith(("http://", "https://", "mailto:", "tel:", "data:", "urn:"))


def _parse_xml(zf: zipfile.ZipFile, name: str, report: EpubQualityGateReport):
    try:
        return ET.fromstring(zf.read(name))
    except KeyError:
        report.add_error(f"缺少文件：{name}")
    except ET.ParseError as exc:
        report.add_error(f"XML/XHTML 无法解析：{name} · {exc}")
    except Exception as exc:
        report.add_error(f"读取失败：{name} · {exc}")
    return None


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _manifest_image_mime_matches(zf: zipfile.ZipFile, name: str, media_type: str) -> bool | None:
    mt = str(media_type or "").lower()
    try:
        head = zf.read(name)[:64]
    except Exception:
        return None
    signatures = _IMAGE_SIGNATURES.get(mt)
    if signatures is not None:
        return any(head.startswith(sig) for sig in signatures)
    if mt == "image/svg+xml":
        stripped = head.lstrip().lower()
        return stripped.startswith(b"<svg") or b"<svg" in stripped
    return None


def _opf_metadata(opf: ET.Element) -> tuple[dict[str, list[ET.Element]], list[ET.Element]]:
    dc: dict[str, list[ET.Element]] = {}
    metas: list[ET.Element] = []
    for element in opf.iter():
        ns = _namespace(element.tag)
        local = _tag_local(element.tag)
        if ns == DC_NS:
            dc.setdefault(local, []).append(element)
        elif local == "meta":
            metas.append(element)
    return dc, metas


def validate_epub_quality(epub_path: str | Path, *, expect_vertical: bool | None = None) -> EpubQualityGateReport:
    report = EpubQualityGateReport()
    report.checks["publication_target"] = STABLE_TARGET
    report.checks["draft_compatibility"] = DRAFT_COMPAT
    path = Path(epub_path)
    if not path.is_file():
        report.add_error(f"EPUB 不存在：{path}")
        return report

    try:
        with zipfile.ZipFile(path) as zf:
            validate_zip(zf)
            infos = zf.infolist()
            names = {info.filename for info in infos}
            report.checks["entry_count"] = len(infos)
            if not infos:
                report.add_error("EPUB ZIP 为空")
                return report

            # EPUBCheck OPF-060 parity: names that differ only after Unicode
            # canonical normalization + full case folding are duplicates from
            # an EPUB/reading-system point of view. This matters especially on
            # macOS, whose filesystem commonly exposes decomposed Unicode.
            normalized_names: dict[str, str] = {}
            normalized_collisions = 0
            space_names = 0
            trailing_dot_names = 0
            non_ascii_names = 0
            for info in infos:
                name = str(info.filename or "")
                normalized = unicodedata.normalize("NFC", name).casefold()
                previous = normalized_names.get(normalized)
                if previous is not None and previous != name:
                    normalized_collisions += 1
                    report.add_error(f"ZIP 路径经 Unicode 规范化/大小写折叠后冲突：{previous!r} / {name!r}")
                else:
                    normalized_names[normalized] = name
                if " " in name:
                    space_names += 1
                    report.add_warning(f"OCF 文件名含空格，旧阅读器兼容性可能较差：{name}")
                if any(part.endswith(".") for part in PurePosixPath(name).parts):
                    trailing_dot_names += 1
                    report.add_error(f"OCF 文件名不允许以句点结尾：{name}")
                if any(ord(ch) > 0x7F for ch in name):
                    non_ascii_names += 1
                    report.add_warning(f"OCF 文件名含非 ASCII 字符，旧工具链兼容性可能较差：{name}")
            report.checks["unicode_normalized_path_collisions"] = normalized_collisions
            report.checks["filenames_with_spaces"] = space_names
            report.checks["filenames_with_trailing_dot"] = trailing_dot_names
            report.checks["non_ascii_filenames"] = non_ascii_names

            # OCF physical container requirements.
            first = infos[0]
            if first.filename != "mimetype":
                report.add_error("mimetype 必须是 ZIP 第一项")
            else:
                if first.compress_type != zipfile.ZIP_STORED:
                    report.add_error("mimetype 必须使用 STORE（不压缩）")
                if first.extra:
                    report.add_error("mimetype 的 ZIP central header 不允许 extra field")
                if first.flag_bits & 0x1:
                    report.add_error("mimetype 不允许 ZIP 加密")
                try:
                    local = _zip_local_header(path, first)
                except Exception as exc:
                    report.add_error(f"mimetype local header 无法读取：{exc}")
                    local = {}
                report.checks["mimetype_local_header_offset"] = local.get("header_offset")
                report.checks["mimetype_local_extra_field_bytes"] = local.get("extra_len")
                report.checks["mimetype_local_compress_type"] = local.get("compression")
                if local:
                    if int(local.get("header_offset", -1)) != 0:
                        report.add_error("mimetype 必须是物理 ZIP 的第一项（local header offset 必须为 0）")
                    if str(local.get("filename") or "") != "mimetype":
                        report.add_error("mimetype local header 文件名与 central directory 不一致")
                    if int(local.get("compression", -1)) != zipfile.ZIP_STORED:
                        report.add_error("mimetype local header 必须声明 STORE（不压缩）")
                    if int(local.get("extra_len", -1)) != 0:
                        report.add_error("mimetype 的 ZIP local header 不允许 extra field")
                    if int(local.get("flag_bits", 0)) & 0x1:
                        report.add_error("mimetype local header 不允许 ZIP 加密")

            invalid_compression = 0
            invalid_extract_version = 0
            encrypted_entries = 0
            for info in infos:
                if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                    invalid_compression += 1
                    report.add_error(f"OCF ZIP 只允许 STORE/DEFLATE：{info.filename}")
                if int(getattr(info, "extract_version", 0) or 0) not in {10, 20, 45}:
                    invalid_extract_version += 1
                    report.add_error(
                        f"OCF ZIP version needed to extract 非法：{info.filename} -> {getattr(info, 'extract_version', '?')}"
                    )
                if info.flag_bits & 0x1:
                    encrypted_entries += 1
                    report.add_error(f"OCF ZIP 不允许使用 ZIP 自带加密：{info.filename}")
            report.checks["ocf_invalid_compression_entries"] = invalid_compression
            report.checks["ocf_invalid_extract_version_entries"] = invalid_extract_version
            report.checks["ocf_encrypted_entries"] = encrypted_entries
            report.checks["mimetype_extra_field_bytes"] = len(first.extra) if first.filename == "mimetype" else None
            try:
                mime = zf.read("mimetype").decode("ascii", errors="replace")
                if mime != MIMETYPE:
                    report.add_error(f"mimetype 内容错误：{mime!r}")
            except KeyError:
                report.add_error("缺少 mimetype")

            container = _parse_xml(zf, CONTAINER_PATH, report)
            opf_path = ""
            if container is not None:
                if _namespace(container.tag) != OCF_NS:
                    report.add_error(
                        "container.xml 命名空间不符合 EPUB OCF："
                        f"{_namespace(container.tag) or '(none)'}"
                    )
                if str(container.attrib.get("version") or "") != "1.0":
                    report.add_error("container.xml 的 version 必须为 1.0")
                rootfiles = [e for e in container.iter() if _tag_local(e.tag) == "rootfile"]
                if not rootfiles:
                    report.add_error("container.xml 没有 rootfile/full-path")
                else:
                    first_rootfile = rootfiles[0]
                    opf_path = str(first_rootfile.attrib.get("full-path") or "").strip()
                    media_type = str(first_rootfile.attrib.get("media-type") or "").strip()
                    if media_type != "application/oebps-package+xml":
                        report.add_error("container.xml rootfile media-type 必须为 application/oebps-package+xml")
                    if not opf_path:
                        report.add_error("container.xml rootfile 缺少 full-path")
            if not opf_path:
                return report
            if opf_path not in names:
                report.add_error(f"container.xml 指向不存在的 OPF：{opf_path}")
                return report

            # Package document.
            opf = _parse_xml(zf, opf_path, report)
            if opf is None:
                return report
            if _namespace(opf.tag) != OPF_NS or _tag_local(opf.tag) != "package":
                report.add_error("Package Document 根元素/命名空间不是 EPUB OPF package")
            version = str(opf.attrib.get("version") or "")
            report.checks["opf_version"] = version
            if version != "3.0":
                report.add_error(f"EPUB 3 Package Document version 应为 3.0，当前为 {version or 'unknown'}")

            dc, meta_elements = _opf_metadata(opf)
            metadata_elements = [e for e in list(opf) if _tag_local(e.tag) == "metadata"]
            if len(metadata_elements) != 1:
                report.add_error(f"OPF package 必须且只能有一个直接子级 metadata，当前 {len(metadata_elements)} 个")
            elif metadata_elements[0].attrib:
                report.add_error("OPF metadata 元素不允许携带属性")
            titles = [_text(e) for e in dc.get("title", []) if _text(e)]
            languages = [_text(e) for e in dc.get("language", []) if _text(e)]
            identifiers = dc.get("identifier", [])
            if not titles:
                report.add_error("OPF metadata 缺少非空 dc:title")
            if not languages:
                report.add_error("OPF metadata 缺少非空 dc:language")
            for language in languages:
                if any(ch.isspace() for ch in language) or "_" in language:
                    report.add_error(f"dc:language 不是规范 BCP47 形式：{language!r}")
            if not identifiers or not any(_text(e) for e in identifiers):
                report.add_error("OPF metadata 缺少非空 dc:identifier")

            unique_identifier = str(opf.attrib.get("unique-identifier") or "").strip()
            unique_identifier_value = ""
            report.checks["unique_identifier_id"] = unique_identifier
            if not unique_identifier:
                report.add_error("OPF package 缺少 unique-identifier")
            else:
                matching = [e for e in identifiers if str(e.attrib.get("id") or "") == unique_identifier]
                if not matching:
                    report.add_error(f"unique-identifier={unique_identifier!r} 没有对应 dc:identifier")
                elif not _text(matching[0]):
                    report.add_error("unique-identifier 对应的 dc:identifier 为空")
                else:
                    unique_identifier_value = _text(matching[0])

            modified_values = [
                _text(e) for e in meta_elements
                if str(e.attrib.get("property") or "").strip() == "dcterms:modified"
            ]
            report.checks["dcterms_modified_count"] = len(modified_values)
            if len(modified_values) != 1:
                report.add_error("EPUB 3 metadata 必须且只能包含一个 dcterms:modified")
            elif not _MODIFIED_RE.match(modified_values[0]):
                report.add_error(f"dcterms:modified 不是 UTC EPUB 日期格式：{modified_values[0]!r}")

            metadata_ids: set[str] = set()
            metadata_by_id: dict[str, ET.Element] = {}
            if metadata_elements:
                for element in metadata_elements[0].iter():
                    metadata_id = str(element.attrib.get("id") or "").strip()
                    if metadata_id:
                        metadata_ids.add(metadata_id)
                        metadata_by_id[metadata_id] = element
            dangling_refines = 0
            invalid_group_positions = 0
            invalid_identifier_types = 0
            for element in meta_elements:
                refines = str(element.attrib.get("refines") or "").strip()
                if refines:
                    if not refines.startswith("#") or refines[1:] not in metadata_ids:
                        dangling_refines += 1
                        report.add_error(f"OPF meta refines 指向不存在的 metadata id：{refines}")
                prop = str(element.attrib.get("property") or "").strip()
                if prop == "group-position" and not re.fullmatch(r"\d+(?:\.\d+)*", _text(element)):
                    invalid_group_positions += 1
                    report.add_error(f"group-position 必须是数字或点分数字序列：{_text(element)!r}")
                if prop == "belongs-to-collection" and not _text(element):
                    report.add_error("belongs-to-collection 不能为空")
                if (
                    prop == "identifier-type"
                    and str(element.attrib.get("scheme") or "").strip() == "onix:codelist5"
                    and refines.startswith("#")
                ):
                    target = metadata_by_id.get(refines[1:])
                    if target is not None and _tag_local(target.tag) == "identifier":
                        compact_identifier = re.sub(r"[^0-9Xx]", "", _text(target))
                        declared_type = _text(element)
                        expected_type = "02" if len(compact_identifier) == 10 else "15" if len(compact_identifier) == 13 else ""
                        if expected_type and declared_type != expected_type:
                            invalid_identifier_types += 1
                            report.add_error(
                                f"ISBN identifier-type 与位数不一致：{_text(target)!r} 应为 ONIX {expected_type}，实际为 {declared_type!r}"
                            )
            report.checks["dangling_metadata_refines"] = dangling_refines
            report.checks["invalid_group_positions"] = invalid_group_positions
            report.checks["invalid_identifier_types"] = invalid_identifier_types

            package_lang = str(opf.attrib.get(f"{{{XML_NS}}}lang") or "").strip()
            if package_lang and languages and package_lang.lower() != languages[0].lower():
                report.add_warning(f"OPF xml:lang={package_lang} 与 dc:language={languages[0]} 不一致")

            manifest: dict[str, dict[str, str]] = {}
            manifest_hrefs: dict[str, str] = {}
            duplicate_manifest_ids = 0
            duplicate_hrefs = 0
            spine_ids: list[str] = []
            spine_linearity: list[str] = []
            spine_ppd = ""
            spine_toc = ""
            for element in opf.iter():
                local = _tag_local(element.tag)
                if local == "item":
                    item_id = str(element.attrib.get("id") or "").strip()
                    href = str(element.attrib.get("href") or "").strip()
                    if not item_id:
                        report.add_error("manifest item 缺少 id")
                        continue
                    if item_id in manifest:
                        duplicate_manifest_ids += 1
                        report.add_error(f"manifest 重复 id：{item_id}")
                        continue
                    if href and href in manifest_hrefs:
                        duplicate_hrefs += 1
                        report.add_error(f"manifest 多个 item 指向同一 href：{href}")
                    manifest_hrefs[href] = item_id
                    manifest[item_id] = {
                        "href": href,
                        "media_type": str(element.attrib.get("media-type") or "").strip(),
                        "properties": str(element.attrib.get("properties") or "").strip(),
                    }
                elif local == "spine":
                    spine_ppd = str(element.attrib.get("page-progression-direction") or "").strip()
                    spine_toc = str(element.attrib.get("toc") or "").strip()
                elif local == "itemref":
                    spine_ids.append(str(element.attrib.get("idref") or "").strip())
                    spine_linearity.append(str(element.attrib.get("linear") or "yes").strip().lower())

            report.checks["manifest_items"] = len(manifest)
            report.checks["spine_items"] = len(spine_ids)
            report.checks["duplicate_manifest_ids"] = duplicate_manifest_ids
            report.checks["duplicate_manifest_hrefs"] = duplicate_hrefs
            if not manifest:
                report.add_error("OPF manifest 为空")
            if not spine_ids:
                report.add_error("OPF spine 为空")
            duplicate_spine_refs = len(spine_ids) - len(set(spine_ids))
            report.checks["duplicate_spine_itemrefs"] = duplicate_spine_refs
            if duplicate_spine_refs:
                report.add_error(f"spine 重复引用同一 manifest item：{duplicate_spine_refs} 处")
            linear_spine_items = sum(1 for value in spine_linearity if value != "no")
            report.checks["linear_spine_items"] = linear_spine_items
            if spine_ids and linear_spine_items == 0:
                report.add_error("spine 没有任何 linear 内容资源")

            nav_ids = [item_id for item_id, item in manifest.items() if "nav" in item.get("properties", "").split()]
            report.checks["nav_item_count"] = len(nav_ids)
            if len(nav_ids) != 1:
                report.add_error(f"EPUB3 manifest 必须且只能有一个 properties=nav 文档，当前 {len(nav_ids)} 个")
            elif manifest.get(nav_ids[0], {}).get("media_type") != "application/xhtml+xml":
                report.add_error("properties=nav 的 manifest item 必须是 application/xhtml+xml")
            if spine_toc and spine_toc not in manifest:
                report.add_error(f"spine toc={spine_toc!r} 没有对应 manifest item")
            elif spine_toc and manifest.get(spine_toc, {}).get("media_type") != "application/x-dtbncx+xml":
                report.add_error("spine toc 必须引用 application/x-dtbncx+xml 资源")

            cover_ids = [
                item_id for item_id, item in manifest.items()
                if "cover-image" in item.get("properties", "").split()
            ]
            report.checks["cover_image_item_count"] = len(cover_ids)
            if len(cover_ids) > 1:
                report.add_error(f"manifest 最多只能声明一个 cover-image，当前 {len(cover_ids)} 个")
            elif cover_ids:
                cover_item = manifest[cover_ids[0]]
                if not str(cover_item.get("media_type") or "").startswith("image/"):
                    report.add_error("cover-image manifest item 必须是图片媒体类型")

            missing_resources: list[str] = []
            xhtml_paths: list[str] = []
            css_paths: list[str] = []
            image_paths: list[tuple[str, str]] = []
            resolved_manifest_paths: set[str] = set()
            for item_id, item in manifest.items():
                href = item.get("href", "")
                media_type = item.get("media_type", "")
                if not href:
                    report.add_error(f"manifest item {item_id} 缺少 href")
                    continue
                if "#" in href:
                    report.add_error(f"manifest item href 不允许带 fragment identifier：{item_id} -> {href}")
                if not media_type:
                    report.add_error(f"manifest item {item_id} 缺少 media-type")
                if _is_external_ref(href):
                    report.add_error(f"Novel Formatter 离线出版配置不允许 manifest 使用远程 href：{href}")
                    continue
                resolved = _resolve_href(opf_path, href)
                resolved_manifest_paths.add(resolved)
                if resolved not in names:
                    missing_resources.append(resolved)
                    continue
                if media_type in {"application/xhtml+xml", "text/html"} or resolved.lower().endswith((".xhtml", ".html", ".htm")):
                    xhtml_paths.append(resolved)
                if media_type == "text/css" or resolved.lower().endswith(".css"):
                    css_paths.append(resolved)
                if media_type.startswith("image/"):
                    image_paths.append((resolved, media_type))
                    match = _manifest_image_mime_matches(zf, resolved, media_type)
                    if match is False:
                        report.add_error(f"图片字节与 manifest media-type 不一致：{resolved} -> {media_type}")
            codec_deflated = 0
            codec_stored = 0
            for resource_name, _media_type in image_paths:
                try:
                    info = zf.getinfo(resource_name)
                except KeyError:
                    continue
                if info.compress_type == zipfile.ZIP_STORED:
                    codec_stored += 1
                elif info.compress_type == zipfile.ZIP_DEFLATED:
                    codec_deflated += 1
            report.checks["codec_resources_stored"] = codec_stored
            report.checks["codec_resources_deflated"] = codec_deflated
            if codec_deflated:
                report.add_warning(
                    f"有 {codec_deflated} 个图片资源被 ZIP 二次 DEFLATE；可改用 STORE 减少打包/解包 CPU"
                )

            for missing in missing_resources[:30]:
                report.add_error(f"manifest 资源不存在：{missing}")
            if len(missing_resources) > 30:
                report.add_error(f"另有 {len(missing_resources)-30} 个 manifest 资源不存在")

            for idref in spine_ids:
                if not idref:
                    report.add_error("spine itemref 缺少 idref")
                    continue
                item = manifest.get(idref)
                if item is None:
                    report.add_error(f"spine 引用不存在的 manifest id：{idref}")
                    continue
                if item.get("media_type") not in {"application/xhtml+xml", "image/svg+xml"}:
                    report.add_error(f"spine 项 {idref} 不是 XHTML/SVG 内容文档：{item.get('media_type')}")

            # EPUB3 does not require NCX, but Novel Formatter intentionally
            # emits one for legacy Kindle/Kobo compatibility. When present it
            # must identify the same publication as the OPF (EPUBCheck NCX-001).
            ncx_ids = [
                item_id for item_id, item in manifest.items()
                if item.get("media_type") == "application/x-dtbncx+xml"
            ]
            report.checks["ncx_item_count"] = len(ncx_ids)
            for ncx_id in ncx_ids[:1]:
                href = manifest[ncx_id].get("href", "")
                if not href or _is_external_ref(href):
                    continue
                ncx_path = _resolve_href(opf_path, href)
                if ncx_path not in names:
                    continue
                ncx_root = _parse_xml(zf, ncx_path, report)
                if ncx_root is None:
                    continue
                ncx_uid = ""
                for element in ncx_root.iter():
                    if _tag_local(element.tag) == "meta" and str(element.attrib.get("name") or "") == "dtb:uid":
                        ncx_uid = str(element.attrib.get("content") or "").strip()
                        break
                report.checks["ncx_uid"] = ncx_uid
                if unique_identifier_value and ncx_uid and ncx_uid != unique_identifier_value:
                    report.add_error(
                        f"NCX dtb:uid 与 OPF unique identifier 不一致：{ncx_uid!r} != {unique_identifier_value!r}"
                    )
                elif unique_identifier_value and not ncx_uid:
                    report.add_error("NCX 缺少 dtb:uid")

            # Detect accidental files copied into the publication tree without
            # a manifest entry. OCF reserved files and the package document are exempt.
            package_dir = str(PurePosixPath(opf_path).parent)
            prefix = (package_dir.rstrip("/") + "/") if package_dir not in {"", "."} else ""
            unmanifested: list[str] = []
            for name in sorted(names):
                if name in {"mimetype", CONTAINER_PATH, opf_path} or name.startswith("META-INF/"):
                    continue
                if prefix and not name.startswith(prefix):
                    continue
                if name not in resolved_manifest_paths:
                    unmanifested.append(name)
            report.checks["unmanifested_resources"] = len(unmanifested)
            for name in unmanifested[:20]:
                report.add_error(f"OPF 目录中存在未列入 manifest 的资源：{name}")

            duplicate_id_count = 0
            ruby_count = rt_count = 0
            image_without_alt = 0
            remote_refs = 0
            external_hyperlinks = 0
            parsed_xhtml: dict[str, tuple[ET.Element, set[str]]] = {}
            pending_refs: list[tuple[str, str, str]] = []
            for name in xhtml_paths:
                root = _parse_xml(zf, name, report)
                if root is None:
                    continue
                if _tag_local(root.tag) != "html" or _namespace(root.tag) != XHTML_NS:
                    report.add_error(f"XHTML 根元素/命名空间错误：{name}")
                html_lang = str(root.attrib.get("lang") or "").strip()
                xml_lang = str(root.attrib.get(f"{{{XML_NS}}}lang") or "").strip()
                if html_lang and xml_lang and html_lang.lower() != xml_lang.lower():
                    report.add_warning(f"XHTML lang/xml:lang 不一致：{name}")
                expected_lang = languages[0] if languages else package_lang
                effective_lang = html_lang or xml_lang
                if expected_lang and effective_lang and expected_lang.lower() != effective_lang.lower():
                    report.add_warning(f"XHTML 语言 {effective_lang} 与 OPF 语言 {expected_lang} 不一致：{name}")
                if not effective_lang:
                    report.add_warning(f"XHTML 缺少 lang/xml:lang：{name}")

                ids: set[str] = set()
                for element in root.iter():
                    local = _tag_local(element.tag)
                    element_id = str(element.attrib.get("id") or "")
                    if element_id:
                        if element_id in ids:
                            duplicate_id_count += 1
                            report.add_error(f"XHTML 重复 id：{name}#{element_id}")
                        ids.add(element_id)
                    if local == "ruby":
                        ruby_count += 1
                    elif local == "rt":
                        rt_count += 1
                    elif local == "img":
                        if "alt" not in element.attrib:
                            image_without_alt += 1
                            report.add_warning(f"图片缺少 alt 属性：{name}")
                    for attr in ("href", "src"):
                        ref = str(element.attrib.get(attr) or "").strip()
                        if not ref:
                            continue
                        if ref.lower().startswith(("http://", "https://")):
                            if local == "a" and attr == "href":
                                external_hyperlinks += 1
                            else:
                                remote_refs += 1
                                report.add_error(f"Novel Formatter 离线出版配置不允许 XHTML 嵌入远程资源：{name} -> {ref}")
                            continue
                        if _is_external_ref(ref):
                            continue
                        pending_refs.append((name, attr, ref))
                parsed_xhtml[name] = (root, ids)

            # Navigation document semantic checks.
            if len(nav_ids) == 1:
                nav_item = manifest.get(nav_ids[0], {})
                nav_path = _resolve_href(opf_path, nav_item.get("href", ""))
                nav_root = parsed_xhtml.get(nav_path, (None, set()))[0]
                if nav_root is not None:
                    toc_navs: list[ET.Element] = []
                    landmark_navs: list[ET.Element] = []
                    for element in nav_root.iter():
                        if _tag_local(element.tag) != "nav":
                            continue
                        epub_type = str(element.attrib.get(f"{{{EPUB_NS}}}type") or "")
                        types = epub_type.split()
                        if "toc" in types:
                            toc_navs.append(element)
                        if "landmarks" in types:
                            landmark_navs.append(element)
                    report.checks["toc_nav_count"] = len(toc_navs)
                    report.checks["landmarks_nav_count"] = len(landmark_navs)
                    if len(toc_navs) != 1:
                        report.add_error(f"nav 文档必须且只能有一个 epub:type=toc，当前 {len(toc_navs)} 个")
                    else:
                        direct_ols = [e for e in list(toc_navs[0]) if _tag_local(e.tag) == "ol"]
                        if len(direct_ols) != 1:
                            report.add_error(f"toc nav 必须且只能有一个直接子级 <ol>，当前 {len(direct_ols)} 个")
                        elif not any(_tag_local(e.tag) == "li" for e in list(direct_ols[0])):
                            report.add_error("toc nav 的 <ol> 至少需要一个 <li> 导航项")
                        else:
                            spine_path_order: dict[str, int] = {}
                            for spine_index, idref in enumerate(spine_ids):
                                item = manifest.get(idref)
                                if not item or _is_external_ref(item.get("href", "")):
                                    continue
                                spine_path_order.setdefault(_resolve_href(opf_path, item.get("href", "")), spine_index)
                            previous_order = -1
                            for anchor in toc_navs[0].iter():
                                if _tag_local(anchor.tag) != "a":
                                    continue
                                href = str(anchor.attrib.get("href") or "").strip()
                                if not href:
                                    continue
                                if _is_external_ref(href):
                                    report.add_error(f"toc nav 不允许链接到远程资源：{href}")
                                    continue
                                target_name, _fragment = _resolve_document_ref(nav_path, href)
                                order = spine_path_order.get(target_name)
                                if order is None:
                                    report.add_error(f"toc nav 目标不在 spine：{href}")
                                    continue
                                if order < previous_order:
                                    report.add_error(f"toc nav 顺序早于前一个 spine 目标：{href}")
                                previous_order = max(previous_order, order)
                    if len(landmark_navs) > 1:
                        report.add_error("nav 文档最多只能有一个 epub:type=landmarks")
                    elif landmark_navs:
                        bodymatter_count = 0
                        spine_path_order = {}
                        for spine_index, idref in enumerate(spine_ids):
                            item = manifest.get(idref)
                            if not item or _is_external_ref(item.get("href", "")):
                                continue
                            spine_path_order.setdefault(_resolve_href(opf_path, item.get("href", "")), spine_index)
                        for element in landmark_navs[0].iter():
                            if _tag_local(element.tag) != "a":
                                continue
                            link_type = str(element.attrib.get(f"{{{EPUB_NS}}}type") or "")
                            if not link_type.strip():
                                report.add_error("landmarks nav 内的 <a> 必须声明 epub:type")
                            href = str(element.attrib.get("href") or "").strip()
                            if href and _is_external_ref(href):
                                report.add_error(f"landmarks nav 不允许链接到远程资源：{href}")
                            if "bodymatter" in link_type.split():
                                bodymatter_count += 1
                                if href and not _is_external_ref(href):
                                    target_name, _fragment = _resolve_document_ref(nav_path, href)
                                    if target_name not in spine_path_order:
                                        report.add_error(f"bodymatter landmark 目标不在 spine：{href}")
                        report.checks["bodymatter_landmarks"] = bodymatter_count
                        if bodymatter_count == 0:
                            report.add_warning("landmarks nav 建议提供 epub:type=bodymatter 入口")

            broken_link_count = 0
            for source_name, attr, ref in pending_refs:
                target_name, fragment = _resolve_document_ref(source_name, ref)
                if target_name not in names:
                    broken_link_count += 1
                    report.add_error(f"XHTML {attr} 引用不存在：{source_name} -> {ref}")
                    continue
                if fragment and target_name in parsed_xhtml:
                    target_ids = parsed_xhtml[target_name][1]
                    if fragment not in target_ids:
                        broken_link_count += 1
                        report.add_error(f"XHTML 锚点不存在：{source_name} -> {ref}")

            css_broken_count = 0
            for css_name in css_paths:
                try:
                    css_text_one = zf.read(css_name).decode("utf-8", errors="replace")
                except Exception:
                    continue
                for match in re.finditer(r"url\(\s*(['\"]?)([^'\")]+)\1\s*\)", css_text_one, re.I):
                    ref = str(match.group(2) or "").strip()
                    if ref.lower().startswith(("http://", "https://")):
                        remote_refs += 1
                        report.add_error(f"Novel Formatter 离线出版配置不允许 CSS 嵌入远程资源：{css_name} -> {ref}")
                        continue
                    if _is_external_ref(ref) or ref.startswith("#"):
                        continue
                    target_name, _fragment = _resolve_document_ref(css_name, ref)
                    if target_name not in names:
                        css_broken_count += 1
                        report.add_error(f"CSS url() 引用不存在：{css_name} -> {ref}")

            report.checks["ruby_count"] = ruby_count
            report.checks["rt_count"] = rt_count
            report.checks["duplicate_ids"] = duplicate_id_count
            report.checks["broken_internal_links"] = broken_link_count
            report.checks["broken_css_resources"] = css_broken_count
            report.checks["images_without_alt"] = image_without_alt
            report.checks["remote_refs"] = remote_refs
            report.checks["external_hyperlinks"] = external_hyperlinks
            if rt_count > ruby_count:
                report.add_error(f"<rt> 数量 {rt_count} 大于 <ruby> 数量 {ruby_count}")

            if expect_vertical is not None:
                css_text = "\n".join(zf.read(name).decode("utf-8", errors="replace") for name in css_paths)
                has_vertical = bool(re.search(r"(?:^|[;{\s])(?:-epub-|-webkit-)?writing-mode\s*:\s*vertical-rl", css_text, re.I))
                report.checks["vertical_css"] = has_vertical
                report.checks["page_progression_direction"] = spine_ppd
                if expect_vertical:
                    if not has_vertical:
                        report.add_error("竖排 EPUB 缺少 writing-mode: vertical-rl")
                    if spine_ppd.lower() != "rtl":
                        report.add_error("竖排 EPUB spine 缺少 page-progression-direction=rtl")
                elif spine_ppd.lower() == "rtl" and not has_vertical:
                    report.add_warning("横排构建却声明了 rtl 翻页方向")
    except zipfile.BadZipFile as exc:
        report.add_error(f"EPUB 不是有效 ZIP：{exc}")
    except Exception as exc:
        report.add_error(f"EPUB 质量检查异常：{exc}")
    return report
