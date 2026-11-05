#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic OCR pipeline benchmark: render, preprocess, OCR, formatter, EPUB.

The benchmark is intentionally *not* part of the pytest suite.  Run it directly::

    .venv/bin/python tools/bench_pipeline.py --label after
    .venv/bin/python tools/bench_pipeline.py --repo /tmp/nf-base --label before
    .venv/bin/python tools/bench_pipeline.py --compare BEFORE.json AFTER.json

Every stage is executed once cold (first call in this process: imports, worker
start-up, first-time initialisation) and ``--runs`` times warm; the report keeps
the cold time and the median of the warm runs.  One extra run per stage is
recorded with :mod:`cProfile` (``<stage>.prof`` plus the top 25 entries by
cumulative time).  PDF rendering happens in worker processes, so its profile
only covers the coordinating parent process.

Stage (c) uses a deterministic stub recognizer installed through the
``adapters.column_ocr_adapter._recognizer_iterator`` hook: column isolation,
crop generation and document assembly run the production code, only the model
inference is replaced.  The report states this explicitly.

End-to-end jobs (``--skip-stages`` runs only these) execute the whole chain
render -> column segmentation -> OCR -> formatter -> EPUB in a fresh child
process per sample: job 1 is the cold start, job 2 the warm start in the same
process.  The render cache is released before each job so stage (a) always
renders.  One more child runs the cold job under cProfile (top 10 cumulative);
``_scanline_connected_ink_components`` is additionally timed by a wrapper that
sums wall time over all threads, because cProfile only sees the main thread.

``--repo`` points the same script at another checkout (for example a
``git worktree`` of the baseline commit), so "before" and "after" share the
exact same input data and measurement code.  ``HOME`` is redirected to a fresh
temporary directory by default so persistent user caches cannot leak between
the two measurements.
"""
from __future__ import annotations

import argparse
import cProfile
import dataclasses
import hashlib
import io
import json
import os
import platform
import pstats
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

THIS_REPO = Path(__file__).resolve().parent.parent
DEFAULT_PDF = THIS_REPO / "tests" / "fixtures" / "bench" / "bench.pdf"
MAX_FIXTURE_BYTES = 1 * 1024 * 1024

SEED = 20261105
TEXT_PAGES = 40
PAGE_W, PAGE_H = 420.0, 595.0  # A5 in PDF points
MARGIN_X, MARGIN_Y = 42.0, 50.0
FONT_SIZE = 11.0
COLUMN_PITCH = FONT_SIZE * 1.75
ILLUSTRATION_BEFORE_TEXT_PAGE = 20  # zero-based text page index

STAGES = ("render", "render_cache_hit", "preprocess", "ocr", "postprocess", "epub")
STAGE_LABELS = {
    "render": "(a) expand_inputs / pdf_to_images",
    "render_cache_hit": "(a') expand_inputs, cache hit",
    "preprocess": "(b) detect_vertical_columns",
    "ocr": "(c) column OCR run (stub engine)",
    "postprocess": "(d) formatter run_pipeline",
    "epub": "(e) build_epub",
}

_RUBY_MARKUP_RE = re.compile(r"｜([^《]+)《([^》]+)》")
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?")

_NUMERALS = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
_CHAPTER_TITLES = ["始まりの朝", "王都への道", "精霊の森", "学園の秘密", "月下の約束", "終わらない夜"]
_RUBY_WORDS = [
    ("魔法", "まほう"), ("精霊", "せいれい"), ("王都", "おうと"),
    ("剣", "つるぎ"), ("学園", "がくえん"), ("黄昏", "たそがれ"),
]
_SUBJECTS = ["彼女", "俺", "少年", "騎士団長", "妹", "先輩", "王女", "老人"]
_OBJECTS = ["古い地図", "銀色の鍵", "小さな手紙", "青い宝石", "壊れた時計", "白い花"]
_ACTIONS = [
    "を静かに見つめていた", "を握りしめて走り出した", "をそっと机の上に置いた",
    "について何も語らなかった", "を探して街を歩き回った",
]
_PLACES = ["窓の外では", "遠くの空で", "石畳の路地に", "城壁の向こうで", "図書館の奥で"]
_EVENTS = ["冷たい風が吹いていた", "鐘の音が響いていた", "誰かの笑い声が聞こえた", "小さな灯りが揺れていた"]
_LINES = [
    "まさか、本当に来るとは思わなかった", "ねえ、少しだけ待ってくれる？",
    "大丈夫だ。俺が何とかする", "……それは、約束できないわ", "行こう。夜が明ける前に",
]
_STUB_SENTENCES = [
    "彼女は静かに笑った。", "遠くで鐘の音が響いていた。", "「行こう。夜が明ける前に」",
    "俺は古い地図を握りしめた。", "……それは、約束できないわ。", "石畳の路地に冷たい風が吹いていた。",
    "王女は窓の外を見つめていた。", "――そのとき、扉が開いた。",
]
_STUB_KANA = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわん"


# ── deterministic book layout ────────────────────────────────────────────────

def _sentence(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.35:
        return f"{rng.choice(_PLACES)}{rng.choice(_EVENTS)}。"
    if roll < 0.75:
        word, reading = rng.choice(_RUBY_WORDS)
        return f"{rng.choice(_SUBJECTS)}は｜{word}《{reading}》の{rng.choice(_OBJECTS)}{rng.choice(_ACTIONS)}。"
    if roll < 0.88:
        return f"{rng.choice(_SUBJECTS)}は息を呑んだ――{rng.choice(_OBJECTS)}が消えていたのだ。"
    return f"……{rng.choice(_SUBJECTS)}は何も言わなかった。"


def _paragraph(rng: random.Random) -> tuple[str, str]:
    if rng.random() < 0.3:
        return f"「{rng.choice(_LINES)}」", "dialogue"
    return "　" + "".join(_sentence(rng) for _ in range(rng.randint(2, 4))), "narration"


def _tokens(markup: str) -> list[tuple[str, str]]:
    """Split markup into (markup, display) units; a ruby word is one unit."""
    out: list[tuple[str, str]] = []
    pos = 0
    for match in _RUBY_MARKUP_RE.finditer(markup):
        out.extend((ch, ch) for ch in markup[pos:match.start()])
        out.append((match.group(0), match.group(1)))
        pos = match.end()
    out.extend((ch, ch) for ch in markup[pos:])
    return out


def layout_book(seed: int = SEED, text_pages: int = TEXT_PAGES) -> list[dict]:
    """Return the ordered page list: cover, text pages and one illustration."""
    rng = random.Random(seed)
    rows = int((PAGE_H - 2 * MARGIN_Y) // FONT_SIZE)
    cols = int((PAGE_W - 2 * MARGIN_X) // COLUMN_PITCH)
    finished: list[list[dict]] = []
    current: list[dict] = []
    unit_id = 0

    def flush() -> None:
        finished.append(list(current))
        current.clear()

    def push_column(column: dict) -> None:
        current.append(column)
        if len(current) >= cols:
            flush()

    def place(markup: str, kind: str) -> None:
        nonlocal unit_id
        unit_id += 1
        column_tokens: list[tuple[str, str]] = []
        used = 0
        for token in _tokens(markup):
            width = len(token[-1])
            if used + width > rows and column_tokens:
                push_column(_column(column_tokens, unit_id, kind))
                column_tokens, used = [], 0
            column_tokens.append(token)
            used += width
        if column_tokens:
            push_column(_column(column_tokens, unit_id, kind))

    chapter = 0
    while len(finished) < text_pages:
        chapter += 1
        if current:
            flush()
        title = _CHAPTER_TITLES[(chapter - 1) % len(_CHAPTER_TITLES)]
        place(f"第{_NUMERALS[min(chapter, 10)]}章　{title}", "heading")
        push_column({"display": "", "markup": "", "unit": 0, "kind": "blank"})
        for index in range(rng.randint(60, 90)):
            if len(finished) >= text_pages:
                break
            if index and index % 25 == 0:
                place("◆", "section")
            markup, kind = _paragraph(rng)
            place(markup, kind)

    text = [{"kind": "text", "columns": columns} for columns in finished[:text_pages]]
    pages = [{"kind": "cover"}]
    pages.extend(text[:ILLUSTRATION_BEFORE_TEXT_PAGE])
    pages.append({"kind": "illustration"})
    pages.extend(text[ILLUSTRATION_BEFORE_TEXT_PAGE:])
    return pages


def _column(tokens: list[tuple[str, str]], unit: int, kind: str) -> dict:
    return {
        "display": "".join(display for _, display in tokens),
        "markup": "".join(markup for markup, _ in tokens),
        "unit": unit,
        "kind": kind,
    }


# ── fixture generation ───────────────────────────────────────────────────────

def _draw_vector_illustration(page, page_no: int, *, title: str | None = None) -> None:
    rng = random.Random(SEED + page_no)
    page.draw_rect(page.rect, color=None, fill=(0.96, 0.94, 0.90))
    for band in range(24):
        y0 = band * PAGE_H / 24
        shade = 0.55 + 0.4 * band / 24
        page.draw_rect((0, y0, PAGE_W, y0 + PAGE_H / 24 + 0.5), color=None, fill=(shade, shade * 0.9, 0.8))
    for _ in range(40):
        x, y, r = rng.uniform(30, PAGE_W - 30), rng.uniform(30, PAGE_H - 30), rng.uniform(6, 40)
        page.draw_circle((x, y), r, color=(0.1, 0.1, 0.2), fill=(rng.random(), rng.random(), rng.random()), width=1.2)
    if title:
        page.insert_text((MARGIN_X, MARGIN_Y + 40), title, fontname="japan", fontsize=24)


def _insert_column(page, x: float, text: str) -> None:
    y = MARGIN_Y + FONT_SIZE
    try:
        page.insert_text((x, y), "\n".join(text), fontname="japan", fontsize=FONT_SIZE, lineheight=1.0)
    except TypeError:
        for row, char in enumerate(text):
            page.insert_text((x, y + row * FONT_SIZE), char, fontname="japan", fontsize=FONT_SIZE)


def generate_pdf(target: Path) -> Path:
    """Write the deterministic benchmark PDF and return its path."""
    import pymupdf

    target.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    for page_no, spec in enumerate(layout_book(), start=1):
        page = document.new_page(width=PAGE_W, height=PAGE_H)
        if spec["kind"] == "cover":
            _draw_vector_illustration(page, page_no, title="ベンチマーク物語")
            continue
        if spec["kind"] == "illustration":
            _draw_vector_illustration(page, page_no)
            continue
        for index, column in enumerate(spec["columns"]):
            if column["display"]:
                x = PAGE_W - MARGIN_X - (index + 1) * COLUMN_PITCH + (COLUMN_PITCH - FONT_SIZE) / 2
                _insert_column(page, x, column["display"])
    fixed_date = "D:20260101000000Z"
    document.set_metadata({
        "title": "Novel Formatter benchmark", "author": "bench_pipeline",
        "producer": "bench_pipeline", "creator": "bench_pipeline",
        "creationDate": fixed_date, "modDate": fixed_date,
    })
    temporary = target.with_suffix(".tmp.pdf")
    document.save(str(temporary), garbage=4, deflate=True, no_new_id=True)
    document.close()
    os.replace(temporary, target)
    return target


def ensure_pdf(requested: Path | None, out_dir: Path) -> tuple[Path, str]:
    if requested is not None:
        if not requested.exists():
            generate_pdf(requested)
            return requested, "generated"
        return requested, "existing"
    if DEFAULT_PDF.exists():
        return DEFAULT_PDF, "existing"
    candidate = generate_pdf(out_dir / "bench.pdf")
    if candidate.stat().st_size <= MAX_FIXTURE_BYTES:
        DEFAULT_PDF.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate, DEFAULT_PDF)
        return DEFAULT_PDF, "generated"
    return candidate, "generated-on-the-fly (exceeds 1 MB, not stored in tests/fixtures)"


def _write_illustration_png(path: Path, page_no: int) -> tuple[int, int]:
    from PIL import Image, ImageDraw

    rng = random.Random(SEED * 3 + page_no)
    size = (600, 850)
    image = Image.new("RGB", size, (250, 248, 240))
    draw = ImageDraw.Draw(image)
    for _ in range(30):
        x, y, r = rng.randint(20, 580), rng.randint(20, 830), rng.randint(8, 60)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")
    image.close()
    return size


def build_fixture_document(image_dir: Path):
    """OCR-like UnifiedDocument matching ``bench.pdf`` (with ruby markup and noise)."""
    from models.document import Block, BlockType, PageInfo, UnifiedDocument

    doc = UnifiedDocument()
    doc.metadata.title = "ベンチマーク物語"
    doc.metadata.author = "ベンチ太郎"
    doc.metadata.language = "ja"
    doc.metadata.writing_direction = "vertical-rl"
    doc.metadata.source_engine = "bench_stub"
    order = 0
    printed_page = 0

    def next_id() -> str:
        return f"bench{order:06d}"

    for page_no, spec in enumerate(layout_book(), start=1):
        if spec["kind"] != "text":
            path = image_dir / f"page_{page_no:03d}.png"
            width, height = _write_illustration_png(path, page_no)
            page_type = BlockType.COVER if spec["kind"] == "cover" else BlockType.ILLUSTRATION
            doc.pages.append(PageInfo(page_no=page_no, page_type=page_type, image_path=str(path),
                                      width=width, height=height))
            last = len(doc.blocks) - 1
            doc.blocks.append(Block(
                type=BlockType.IMAGE_REF, image_path=str(path),
                image_anchor=f"block_{last}" if last >= 0 else "start",
                page=page_no, reading_order=order, id=next_id(),
            ))
            order += 1
            continue
        printed_page += 1
        doc.pages.append(PageInfo(page_no=page_no, page_type=BlockType.PARAGRAPH,
                                  image_path=f"page_{page_no:03d}.png"))
        groups: list[list[dict]] = []
        for column in spec["columns"]:
            if not column["markup"]:
                continue
            if groups and groups[-1][-1]["unit"] == column["unit"]:
                groups[-1].append(column)
            else:
                groups.append([column])
        texts = ["".join(column["markup"] for column in group) for group in groups]
        texts.append(str(printed_page))
        for in_page, text in enumerate(texts):
            doc.blocks.append(Block(
                type=BlockType.PARAGRAPH, text=text, ocr_raw=text,
                page=page_no, page_index=page_no - 1, page_number=page_no,
                order_in_page=in_page, reading_order=order, confidence=0.97,
                text_direction="vertical", id=next_id(),
            ))
            order += 1
    return doc


# ── output signatures used for equivalence checks ────────────────────────────

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def page_signatures(paths: list[str]) -> list[dict]:
    from PIL import Image

    rows = []
    for path in paths:
        with Image.open(path) as image:
            rows.append({
                "name": Path(path).name, "mode": image.mode, "size": list(image.size),
                "pixels_sha256": _sha(image.tobytes()),
            })
    return rows


def _column_signature(column: Any) -> Any:
    if dataclasses.is_dataclass(column):
        return dataclasses.asdict(column)
    return repr(column)


def document_signature(doc) -> dict:
    return {
        "blocks": [
            {
                "type": getattr(block.type, "value", str(block.type)),
                "text": block.text,
                "page": block.page,
                "chapter_index": block.chapter_index,
                "image": Path(block.image_path).name if block.image_path else "",
            }
            for block in doc.blocks
        ],
        "toc": [
            {"title": entry.title, "chapter_index": entry.chapter_index, "block_index": entry.block_index}
            for entry in doc.toc
        ],
    }


def epub_signature(path: Path) -> dict:
    members = {}
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            data = archive.read(name)
            if name.endswith((".opf", ".xhtml", ".html", ".css", ".ncx", ".xml")) or name == "mimetype":
                text = data.decode("utf-8", errors="replace")
                text = _DATE_RE.sub("DATE", _UUID_RE.sub("UUID", text))
                data = text.encode("utf-8")
            members[name] = _sha(data)
    return members


def _json_sha(payload: Any) -> str:
    return _sha(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))


# ── stub OCR engine ──────────────────────────────────────────────────────────

def install_stub_recognizer() -> Callable[[], None]:
    """Replace model inference with a pixel-hash keyed deterministic answer."""
    import adapters.column_ocr_adapter as column_ocr
    from PIL import Image

    def stub_iterator(engine, image_paths, manifest_path, *, shortcut_name="ExtractText",
                      cancel_check=None, verbose=True, engine_options=None, input_metadata=None):
        del engine, manifest_path, shortcut_name, verbose, engine_options, input_metadata
        for path in image_paths:
            if cancel_check is not None and cancel_check():
                return
            with Image.open(path) as image:
                gray = image.convert("L")
                darkest = min(gray.getextrema())
                digest = hashlib.sha256(gray.tobytes()).digest()
                gray.close()
            if darkest > 200:
                yield str(path), [], None
                continue
            # Unique per crop (like real column text) so assembly never dedupes it.
            sentence = _STUB_SENTENCES[int.from_bytes(digest[:4], "big") % len(_STUB_SENTENCES)]
            tail = "".join(_STUB_KANA[value % len(_STUB_KANA)] for value in digest[4:16])
            text = f"{sentence.rstrip('。')}{tail}。"
            yield str(path), [{"text": text, "confidence": 0.95}], None

    original = column_ocr._recognizer_iterator
    column_ocr._recognizer_iterator = stub_iterator

    def restore() -> None:
        column_ocr._recognizer_iterator = original

    return restore


# ── measurement ──────────────────────────────────────────────────────────────

def _profile_text(profile: cProfile.Profile, repo: Path) -> str:
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(25)
    return stream.getvalue().replace(str(repo) + os.sep, "").replace(sys.prefix + os.sep, "<venv>/")


def measure(name: str, fn: Callable[[Any], Any], setup: Callable[[], Any] | None,
            runs: int, profile_dir: Path, repo: Path) -> tuple[dict, Any]:
    times: list[float] = []
    result = None
    for _ in range(runs + 1):
        state = setup() if setup is not None else None
        start = time.perf_counter()
        result = fn(state)
        times.append(time.perf_counter() - start)
    state = setup() if setup is not None else None
    profile = cProfile.Profile()
    profile.enable()
    fn(state)
    profile.disable()
    profile_path = profile_dir / f"{name}.prof"
    profile.dump_stats(str(profile_path))
    warm = times[1:]
    return {
        "label": STAGE_LABELS[name],
        "cold_s": round(sum(times[:1]), 6),
        "warm_runs_s": [round(value, 6) for value in warm],
        "warm_median_s": round(statistics.median(warm), 6),
        "profile_file": str(profile_path),
        "profile_top25": _profile_text(profile, repo),
    }, result


def _git_head(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def run_benchmark(args: argparse.Namespace) -> dict:
    repo = Path(args.repo).resolve()
    out_dir = Path(args.out_dir).resolve()
    profile_dir = out_dir / "profiles"
    artifacts = out_dir / "artifacts"
    for directory in (profile_dir, artifacts):
        directory.mkdir(parents=True, exist_ok=True)
    pdf, pdf_origin = ensure_pdf(Path(args.pdf).resolve() if args.pdf else None, out_dir)

    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import fitz  # noqa: F401  (production code imports this name)
    import PIL
    from adapters.column_ocr_adapter import detect_vertical_columns, run as column_run
    from adapters.pdf_input import expand_inputs, release_pdf_caches
    from builder.epub_builder import build_epub
    from engine.formatter import run_pipeline
    from models.document import UnifiedDocument
    from utils.session_temp import cleanup_session_temp

    stages: dict[str, dict] = {}
    dpi = int(args.dpi)

    def render(_state):
        return expand_inputs([str(pdf)], dpi=dpi)

    stages["render"], pages = measure(
        "render", render, lambda: release_pdf_caches([str(pdf)]), args.runs, profile_dir, repo)
    stages["render_cache_hit"], pages = measure(
        "render_cache_hit", render, None, args.runs, profile_dir, repo)
    pages = list(pages)
    page_rows = page_signatures(pages)
    (artifacts / "pages.json").write_text(json.dumps(page_rows, indent=1), encoding="utf-8")

    layout = layout_book()
    text_pages = [path for path, spec in zip(pages, layout) if spec["kind"] == "text"]
    overrides = {index: spec["kind"] for index, spec in enumerate(layout, start=1) if spec["kind"] != "text"}
    overrides = {page: ("cover" if kind == "cover" else "illustration") for page, kind in overrides.items()}

    def preprocess(_state):
        return [[_column_signature(column) for column in detect_vertical_columns(path)] for path in text_pages]

    stages["preprocess"], columns = measure("preprocess", preprocess, None, args.runs, profile_dir, repo)
    columns_sha = _json_sha(columns)

    restore = install_stub_recognizer()
    try:
        def ocr(_state):
            return column_run(recognition_engine="apple_vision", input_paths=list(pages),
                              page_overrides=dict(overrides), verbose=False)

        stages["ocr"], ocr_doc = measure("ocr", ocr, None, args.runs, profile_dir, repo)
    finally:
        restore()
    ocr_signature = document_signature(ocr_doc)
    (artifacts / "ocr_output.json").write_text(
        json.dumps(ocr_signature, ensure_ascii=False, indent=1), encoding="utf-8")

    fixture = build_fixture_document(artifacts / "images")
    fixture_dict = fixture.to_dict()
    (artifacts / "formatter_input.json").write_text(fixture.to_json(), encoding="utf-8")
    repo_root = Path(tempfile.mkdtemp(prefix="nf-bench-repo-"))
    counter = {"n": 0}

    def fresh_document():
        counter["n"] += 1
        return UnifiedDocument.from_dict(json.loads(json.dumps(fixture_dict))), repo_root / f"r{counter['n']}"

    def postprocess(state):
        document, repo_path = state
        return run_pipeline(document, verbose=False, repo_path=str(repo_path))

    stages["postprocess"], formatted = measure(
        "postprocess", postprocess, fresh_document, args.runs, profile_dir, repo)
    formatted_signature = document_signature(formatted)
    (artifacts / "formatter_output.json").write_text(
        json.dumps(formatted_signature, ensure_ascii=False, indent=1), encoding="utf-8")

    epub_path = artifacts / "bench.epub"

    def epub(_state):
        build_epub(formatted, str(epub_path), vertical=True, verbose=False)
        return epub_path

    stages["epub"], _ = measure("epub", epub, None, args.runs, profile_dir, repo)
    epub_members = epub_signature(epub_path)
    shutil.rmtree(repo_root, ignore_errors=True)
    release_pdf_caches([str(pdf)])
    cleanup_session_temp()

    return {
        "meta": {
            "label": args.label,
            "repo": str(repo),
            "git_head": _git_head(repo),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "pymupdf": getattr(sys.modules["fitz"], "VersionBind", "?"),
            "pillow": PIL.__version__,
            "pdf": str(pdf),
            "pdf_origin": pdf_origin,
            "pdf_bytes": pdf.stat().st_size,
            "pages_total": len(pages),
            "text_pages": len(text_pages),
            "dpi": dpi,
            "runs": args.runs,
            "workers_env": os.environ.get("NOVEL_FORMATTER_WORKERS", "default"),
            "engine": "stub recognizer via column_ocr_adapter._recognizer_iterator "
                      "(Apple Vision unavailable: tools/apple_vision_helper/AppleVisionOCRHelper.swift "
                      "missing and no cached helper binary); the stub bypasses the recognizer session cache",
            "home_isolated": bool(args.isolate_home),
        },
        "stages": stages,
        "outputs": {
            "pages": page_rows,
            "pages_sha256": _json_sha(page_rows),
            "columns_sha256": columns_sha,
            "ocr_sha256": _json_sha(ocr_signature),
            "formatter_sha256": _json_sha(formatted_signature),
            "epub_members": epub_members,
        },
    }


# ── end-to-end jobs (cold = first job in a process, warm = second job) ───────

E2E_STAGES = ("render", "segmentation", "ocr", "postprocess", "epub", "total")
E2E_LABELS = {
    "render": "(a) expand_inputs / pdf_to_images",
    "segmentation": "(b) column preprocessing / segmentation",
    "ocr": "(c) OCR (stub engine)",
    "postprocess": "(d) formatter run_pipeline",
    "epub": "(e) build_epub",
    "total": "end-to-end",
}
PROFILE_SHARE_FUNCTIONS = (
    "expand_inputs", "pdf_to_images", "_scanline_connected_ink_components",
    "detect_vertical_columns", "run_pipeline", "build_epub",
)


def _e2e_setup(args: argparse.Namespace):
    repo = Path(args.repo).resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    started = time.perf_counter()
    import fitz  # noqa: F401
    import adapters.column_ocr_adapter as column_ocr
    from adapters.pdf_input import expand_inputs, release_pdf_caches
    from builder.epub_builder import build_epub
    from engine.formatter import run_pipeline
    imports_s = time.perf_counter() - started

    pdf = Path(args.pdf).resolve()
    dpi = int(args.dpi)
    work = Path(tempfile.mkdtemp(prefix="nf-e2e-"))
    layout = layout_book()
    overrides = {
        index: ("cover" if spec["kind"] == "cover" else "illustration")
        for index, spec in enumerate(layout, start=1) if spec["kind"] != "text"
    }
    scan = {"s": 0.0, "calls": 0, "available": False}
    original = getattr(column_ocr, "_scanline_connected_ink_components", None)
    if callable(original):
        scan["available"] = True
        lock = threading.Lock()

        def timed(mask, *a, **kw):
            t = time.perf_counter()
            try:
                return original(mask, *a, **kw)
            finally:
                elapsed = time.perf_counter() - t
                with lock:
                    scan["s"] += elapsed
                    scan["calls"] += 1

        column_ocr._scanline_connected_ink_components = timed
    install_stub_recognizer()

    def job(index: int) -> dict:
        release_pdf_caches([str(pdf)])
        scan_before = scan["s"]
        marks: dict[str, float] = {}

        def phase(name, *_args, **_kwargs):
            if name == "split":
                marks["split_end"] = time.perf_counter()

        t0 = time.perf_counter()
        pages = list(expand_inputs([str(pdf)], dpi=dpi))
        t1 = time.perf_counter()
        doc = column_ocr.run(recognition_engine="apple_vision", input_paths=list(pages),
                             page_overrides=dict(overrides), verbose=False, phase_callback=phase)
        t2 = time.perf_counter()
        ocr_signature = document_signature(doc)
        t2b = time.perf_counter()
        formatted = run_pipeline(doc, verbose=False, repo_path=str(work / f"repo{index}"))
        t3 = time.perf_counter()
        build_epub(formatted, str(work / f"job{index}.epub"), vertical=True, verbose=False)
        t4 = time.perf_counter()
        split_end = marks.get("split_end", t1)
        return {
            "render": t1 - t0,
            "segmentation": split_end - t1,
            "ocr": t2 - split_end,
            "postprocess": t3 - t2b,
            "epub": t4 - t3,
            "total": (t4 - t0) - (t2b - t2),
            "scanline_s": scan["s"] - scan_before,
            "split_phase_seen": "split_end" in marks,
            "pages": len(pages),
            "ocr_sha256": _json_sha(ocr_signature),
            "ocr_text": "\n".join(block["text"] or "" for block in ocr_signature["blocks"]),
        }

    return job, imports_s, scan, work, release_pdf_caches, pdf


def e2e_child(args: argparse.Namespace) -> int:
    job, imports_s, scan, work, release, pdf = _e2e_setup(args)
    result: dict[str, Any] = {"imports_s": imports_s, "scanline_available": scan["available"]}
    if args.e2e_profile:
        profile = cProfile.Profile()
        started = time.perf_counter()
        profile.enable()
        cold = job(1)
        profile.disable()
        wall = time.perf_counter() - started
        profile.dump_stats(str(Path(args.e2e_profile).with_suffix(".prof")))
        stats = pstats.Stats(profile)
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(10)
        repo = Path(args.repo).resolve()
        cumulative = {}
        for (filename, _line, function), (_cc, _nc, _tt, ct, _callers) in stats.stats.items():
            if function in PROFILE_SHARE_FUNCTIONS and str(repo) in str(Path(filename).resolve()):
                cumulative[function] = max(cumulative.get(function, 0.0), ct)
        result.update({
            "profile_wall_s": wall,
            "profile_top10": stream.getvalue().replace(str(repo) + os.sep, "").replace(sys.prefix + os.sep, "<venv>/"),
            "profile_cumulative_s": cumulative,
            "profile_job": {k: v for k, v in cold.items() if k != "ocr_text"},
        })
    else:
        result["cold"] = job(1)
        result["warm"] = job(2)
    release([str(pdf)])
    shutil.rmtree(work, ignore_errors=True)
    Path(args.e2e_child).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def run_e2e(args: argparse.Namespace, out_dir: Path, pdf: Path) -> dict:
    script = str(Path(__file__).resolve())
    samples = []

    def child(tag: str, extra: list[str]) -> dict:
        target = out_dir / f"e2e_{tag}.json"
        home = out_dir / f"home_{tag}"
        home.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        if args.isolate_home:
            env["HOME"] = str(home)
        command = [sys.executable, script, "--repo", str(Path(args.repo).resolve()), "--pdf", str(pdf),
                   "--dpi", str(args.dpi), "--e2e-child", str(target), *extra]
        subprocess.run(command, check=True, env=env)
        return json.loads(target.read_text(encoding="utf-8"))

    for run_index in range(int(args.runs)):
        samples.append(child(f"run{run_index + 1}", []))
    profiled = child("profile", ["--e2e-profile", str(out_dir / "profiles" / "e2e_cold")])

    def median_of(kind: str, stage: str) -> float:
        return round(statistics.median(sample[kind][stage] for sample in samples), 6)

    texts = {sample[kind]["ocr_text"] for sample in samples for kind in ("cold", "warm")}
    stages = {
        stage: {
            "label": E2E_LABELS[stage],
            "cold_runs_s": [round(sample["cold"][stage], 6) for sample in samples],
            "warm_runs_s": [round(sample["warm"][stage], 6) for sample in samples],
            "cold_median_s": median_of("cold", stage),
            "warm_median_s": median_of("warm", stage),
        }
        for stage in E2E_STAGES
    }
    scan_cold = statistics.median(sample["cold"]["scanline_s"] for sample in samples)
    scan_warm = statistics.median(sample["warm"]["scanline_s"] for sample in samples)
    wall = profiled["profile_wall_s"]
    return {
        "stages": stages,
        "imports_median_s": round(statistics.median(sample["imports_s"] for sample in samples), 6),
        "scanline_available": all(sample["scanline_available"] for sample in samples),
        "scanline_cold_median_s": round(scan_cold, 6),
        "scanline_warm_median_s": round(scan_warm, 6),
        "scanline_share_cold": round(scan_cold / stages["total"]["cold_median_s"], 4),
        "render_share_cold": round(stages["render"]["cold_median_s"] / stages["total"]["cold_median_s"], 4),
        "split_phase_seen": all(sample["cold"]["split_phase_seen"] for sample in samples),
        "ocr_sha256": sorted({sample[kind]["ocr_sha256"] for sample in samples for kind in ("cold", "warm")}),
        "ocr_text_stable_across_runs": len(texts) == 1,
        "ocr_text": min(texts),
        "profile": {
            "wall_s": round(wall, 6),
            "top10_cumulative": profiled["profile_top10"],
            "cumulative_s": {k: round(v, 6) for k, v in profiled["profile_cumulative_s"].items()},
            "share_of_wall": {k: round(v / wall, 4) for k, v in profiled["profile_cumulative_s"].items()},
            "scanline_timer_s": round(profiled["profile_job"]["scanline_s"], 6),
            "scanline_timer_share": round(profiled["profile_job"]["scanline_s"] / wall, 4),
            "note": "cProfile only sees the main thread: spawn render workers and the "
                    "prepare_job thread pool are counted as waiting time of their callers; "
                    "the scanline timer sums wall time over all threads.",
        },
    }


def format_e2e(report: dict) -> str:
    e2e = report.get("e2e")
    if not e2e:
        return ""
    lines = [f"{'end-to-end stage':<40} {'cold med, s':>12} {'warm med, s':>12}"]
    for stage in E2E_STAGES:
        row = e2e["stages"][stage]
        lines.append(f"{row['label']:<40} {row['cold_median_s']:>12.3f} {row['warm_median_s']:>12.3f}")
    lines.append(f"imports {e2e['imports_median_s']:.3f}s  scanline cold {e2e['scanline_cold_median_s']:.3f}s "
                 f"({e2e['scanline_share_cold']:.1%})  render share cold {e2e['render_share_cold']:.1%}")
    return "\n".join(lines)


# ── reporting ────────────────────────────────────────────────────────────────

def format_table(report: dict) -> str:
    meta = report["meta"]
    lines = [
        f"{meta['label']} @ {meta['git_head']}  pages={meta['pages_total']} (text {meta['text_pages']})  "
        f"dpi={meta['dpi']}  cpu={meta['cpu_count']}  workers={meta['workers_env']}  runs={meta['runs']}",
        f"engine: {meta['engine']}",
        f"{'stage':<40} {'cold, s':>10} {'warm median, s':>15}",
    ]
    for name in STAGES:
        stage = report["stages"].get(name)
        if stage:
            lines.append(f"{stage['label']:<40} {stage['cold_s']:>10.3f} {stage['warm_median_s']:>15.3f}")
    return "\n".join(lines)


def compare_reports(before: dict, after: dict) -> tuple[str, bool]:
    lines = [
        f"{'stage':<40} {'before cold':>11} {'after cold':>11} {'x':>6} "
        f"{'before warm':>12} {'after warm':>11} {'x':>6}",
    ]
    for name in STAGES:
        b, a = before["stages"].get(name), after["stages"].get(name)
        if not b or not a:
            continue
        cold_x = b["cold_s"] / a["cold_s"] if a["cold_s"] else float("inf")
        warm_x = b["warm_median_s"] / a["warm_median_s"] if a["warm_median_s"] else float("inf")
        lines.append(
            f"{b['label']:<40} {b['cold_s']:>11.3f} {a['cold_s']:>11.3f} {cold_x:>6.2f} "
            f"{b['warm_median_s']:>12.3f} {a['warm_median_s']:>11.3f} {warm_x:>6.2f}"
        )
    equal = True
    for key in ("pages_sha256", "columns_sha256", "ocr_sha256", "formatter_sha256", "epub_members"):
        same = before["outputs"].get(key) == after["outputs"].get(key)
        equal = equal and same
        lines.append(f"equivalence {key:<18} {'identical' if same else 'DIFFERENT'}")
    eb, ea = before.get("e2e"), after.get("e2e")
    if eb and ea:
        lines.append(f"{'end-to-end stage':<40} {'before cold':>11} {'after cold':>11} {'x':>6} "
                     f"{'before warm':>12} {'after warm':>11} {'x':>6}")
        for name in E2E_STAGES:
            b, a = eb["stages"][name], ea["stages"][name]
            cx = b["cold_median_s"] / a["cold_median_s"] if a["cold_median_s"] else float("inf")
            wx = b["warm_median_s"] / a["warm_median_s"] if a["warm_median_s"] else float("inf")
            lines.append(f"{b['label']:<40} {b['cold_median_s']:>11.3f} {a['cold_median_s']:>11.3f} {cx:>6.2f} "
                         f"{b['warm_median_s']:>12.3f} {a['warm_median_s']:>11.3f} {wx:>6.2f}")
        same_text = eb.get("ocr_text") == ea.get("ocr_text")
        equal = equal and same_text
        lines.append(f"equivalence {'e2e_ocr_text':<18} {'identical' if same_text else 'DIFFERENT'}")
    return "\n".join(lines), equal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="".join(__doc__.strip().split("\n", 1)[:1]))
    parser.add_argument("--repo", default=str(THIS_REPO), help="checkout to benchmark (default: this repo)")
    parser.add_argument("--pdf", default=None, help="input PDF (generated if missing)")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3, help="warm runs per stage (median reported)")
    parser.add_argument("--label", default="current")
    parser.add_argument("--out-dir", default=None, help="report/profile directory")
    parser.add_argument("--no-isolate-home", dest="isolate_home", action="store_false",
                        help="keep the real HOME (persistent user caches may skew results)")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE_JSON", "AFTER_JSON"))
    parser.add_argument("--generate-only", action="store_true", help="only (re)generate the fixture PDF")
    parser.add_argument("--skip-stages", action="store_true", help="only run the end-to-end jobs")
    parser.add_argument("--skip-e2e", action="store_true", help="only run the per-stage loop")
    parser.add_argument("--e2e-child", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--e2e-profile", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.e2e_child:
        return e2e_child(args)
    if args.compare:
        before, after = (json.loads(Path(path).read_text(encoding="utf-8")) for path in args.compare)
        text, equal = compare_reports(before, after)
        print(text)
        return 0 if equal else 1
    if args.generate_only:
        target = Path(args.pdf).resolve() if args.pdf else DEFAULT_PDF
        generate_pdf(target)
        print(f"{target} {target.stat().st_size} bytes")
        return 0

    if args.out_dir is None:
        args.out_dir = str(Path(tempfile.gettempdir()) / f"nf-bench-{args.label}")
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    if args.isolate_home:
        home = out_dir / "home"
        home.mkdir()
        os.environ["HOME"] = str(home)

    if args.skip_stages:
        pdf, origin = ensure_pdf(Path(args.pdf).resolve() if args.pdf else None, out_dir)
        (out_dir / "profiles").mkdir(parents=True, exist_ok=True)
        report = {"meta": {"label": args.label, "repo": str(Path(args.repo).resolve()),
                           "git_head": _git_head(Path(args.repo).resolve()), "pdf": str(pdf),
                           "pdf_origin": origin, "dpi": int(args.dpi), "runs": args.runs,
                           "cpu_count": os.cpu_count(), "platform": platform.platform(),
                           "workers_env": os.environ.get("NOVEL_FORMATTER_WORKERS", "default")},
                  "stages": {}, "outputs": {}}
    else:
        report = run_benchmark(args)
    if not args.skip_e2e:
        report["e2e"] = run_e2e(args, out_dir, Path(report["meta"]["pdf"]))
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    table = format_table(report) if report["stages"] else report["meta"]["label"]
    table = table + "\n" + format_e2e(report)
    profiles = "\n\n".join(
        f"## {stage['label']}\n{stage['profile_top25']}" for stage in report["stages"].values()
    )
    if report.get("e2e"):
        profiles = f"## end-to-end cold job, top 10 cumulative\n{report['e2e']['profile']['top10_cumulative']}\n\n" + profiles
    (out_dir / "report.md").write_text(f"```\n{table}\n```\n\n{profiles}\n", encoding="utf-8")
    print(table)
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
