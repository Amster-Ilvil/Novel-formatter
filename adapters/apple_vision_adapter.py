#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apple Vision OCR 适配器
基于 ocr_via_shortcuts.py，通过 macOS 快捷指令调用 Live Text 引擎。

改造重点：
    - 原脚本直接输出 docx；本适配器输出 UnifiedDocument（统一文档模型）
    - 原脚本在单遍处理中同时做分类+输出；本适配器分两遍：
        第一遍：OCR + 页面分类（cover/illustration/text/blank）
        第二遍：构建 UnifiedDocument blocks
    - 页面分类结果可被 Page Manager UI 的手动标注覆盖（传入 page_overrides）
    - 输出 JSON 可直接送入 Novel Formatter Engine

前置条件（同原脚本）：
    1. 打开"快捷指令"App，新建快捷指令，命名为 ExtractText
    2. 添加动作"从图像中提取文字"，输入设为"快捷指令输入"
    3. 保存，用命令测试：shortcuts run ExtractText -i /图片路径.jpg

依赖：
    pip install python-docx   # 如需同时输出 docx

用法（命令行）：
    python apple_vision_adapter.py /图片文件夹 output.json
    python apple_vision_adapter.py /图片文件夹 output.json --overrides overrides.json

overrides.json 格式（Page Manager UI 导出）：
    {"1": "cover", "5": "blank", "28": "illustration"}
    键为页码字符串，值为 BlockType 字符串
"""

from __future__ import annotations

import sys
import os
import re
import glob
import json
import subprocess
import argparse
import threading
from collections import Counter
from pathlib import Path

# 将项目根目录加入 path，方便直接运行
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import (
    UnifiedDocument, Block, BlockType, PageInfo, BoundingBox, Metadata
)
from engine.page_ocr_policy import should_skip_page_ocr
from adapters.ocr_profiles import (
    apply_profile_metadata, auto_classify_pages as auto_classify_pages_profile,
    get_ocr_profile, is_chapter_title, is_chinese_horizontal, normalize_lines,
    normalize_ocr_mode,
)

# ── 配置常量 ──────────────────────────────────────────────────────────────────

SHORTCUT_NAME = "ExtractText"   # 快捷指令名称，与你在 Shortcuts App 中创建的保持一致

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.heic', '.tif', '.tiff', '.bmp', '.gif')

# Multi-model column OCR can ask several model threads to prepare the exact same
# fixed-region page at the same time.  Protect each destination so 400-page books
# decode/encode every page once instead of once per OCR model (or, worse, race two
# Pillow writers against the same PNG).
_CROP_OUTPUT_LOCKS: dict[str, threading.Lock] = {}
_CROP_OUTPUT_LOCKS_GUARD = threading.Lock()


def _crop_output_lock(path: Path) -> threading.Lock:
    key = str(path.expanduser().resolve())
    with _CROP_OUTPUT_LOCKS_GUARD:
        lock = _CROP_OUTPUT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _CROP_OUTPUT_LOCKS[key] = lock
        return lock

# 页眉/页脚识别
HEADER_MAX_LEN = 25         # 超过此长度的行不视为重复页眉
HEADER_THRESHOLD_RATIO = 0.5   # 出现在超过此比例的页面中才视为页眉

# 页面分类阈值
MIN_CONTENT_CHARS = 30      # 少于此字数 → 非正文（插图/空白）
MIN_PUNCT_COUNT   = 5       # 少于此标点数 → 非正文
BLANK_THRESHOLD   = 5       # 少于此字数 → 空白页

# 竖排日文常用标点（用于正文判定）
PUNCT_CHARS = set('。、「」『』,.!?！？…・：；')

# 页码正则（1~6位纯数字行）
PAGE_NUMBER_RE = re.compile(r'^[\d\s]{1,6}$')

# 章节标题正则（序章、第X章、幕間、後記、エピローグ 等）
# フロローグ：竖排文字识别常见的"プロローグ"半浊点丢失错读，一并纳入识别。
JP_CHAPTER_NUMBER = r'[一二三四五六七八九十百千〇零\d０-９]+'
CHAPTER_UNIT = r'[章話節巻回幕篇編]'
CHAPTER_CONTINUATION = r'(?:は|が|を|に|で|と|も|の|です|だ|という)'
CHAPTER_RE = re.compile(
    rf'^(序章|終章|プロローグ|フロローグ|ブロローグ|エピローグ|後記|あとがき|'
    rf'幕間(?:[\s　:：・—―-].*)?|'
    rf'第[\s　]*{JP_CHAPTER_NUMBER}[\s　]*{CHAPTER_UNIT}(?!{CHAPTER_CONTINUATION})|'
    rf'{JP_CHAPTER_NUMBER}[\s　]*[話章節回](?=$|[\s　:：・—―「『【（(]|前編|後編|上編|中編|下編)|'
    rf'(?:Chapter|Episode|EP)[\s　.．_-]*[\d０-９]+)',
    re.IGNORECASE
)


# ── OCR 调用 ──────────────────────────────────────────────────────────────────

def _estimate_mask_background(image) -> tuple[int, int, int]:
    """Estimate a light paper colour for fixed-region masking."""
    rgb = image.convert("RGB")
    try:
        width, height = rgb.size
        samples: list[tuple[int, int, int]] = []
        step_x = max(1, width // 36)
        step_y = max(1, height // 36)
        border = max(2, round(min(width, height) * 0.03))
        for y in range(0, height, step_y):
            for x in range(0, width, step_x):
                if x < border or x >= width - border or y < border or y >= height - border:
                    pixel = rgb.getpixel((x, y))
                    if sum(pixel) >= 570:
                        samples.append(pixel)
        if not samples:
            return (255, 255, 255)
        samples.sort(key=sum)
        keep = samples[len(samples) // 2:]
        return tuple(
            round(sum(pixel[channel] for pixel in keep) / len(keep))
            for channel in range(3)
        )
    finally:
        if rgb is not image:
            rgb.close()


def crop_for_ocr(image_path: str, crop_top: float = 0.0, crop_bottom: float = 0.0,
                  crop_rect: tuple[float, float, float, float] | None = None,
                  out_dir: str | None = None, *, reuse_existing: bool = False) -> str:
    """Mask everything outside the selected OCR region without changing canvas size.

    In multi-model mode every OCR engine consumes byte-identical fixed-region
    pages.  ``reuse_existing`` plus the per-destination lock makes that shared
    cache safe even when model 1/2 start in parallel.
    """
    if crop_rect is None and crop_top <= 0 and crop_bottom <= 0:
        return image_path

    try:
        from PIL import Image
    except ImportError:
        print("  ⚠️  未安装 Pillow，无法生成 OCR 区域掩膜，将使用原图: pip3 install pillow")
        return image_path

    src = Path(image_path)
    out_dir_path = Path(out_dir) if out_dir else src.parent / "_ocr_crop"
    out_dir_path.mkdir(parents=True, exist_ok=True)
    dest = out_dir_path / src.name

    # Always serialize writers for the same destination.  The fast existence
    # check is repeated inside the lock because another model may finish while
    # this caller is waiting.
    with _crop_output_lock(dest):
        if reuse_existing and dest.exists() and dest.stat().st_size > 0:
            return str(dest)

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        try:
            width, height = image.size
            if crop_rect is not None:
                x0, y0, x1, y1 = crop_rect
                x0 = max(0.0, min(1.0, float(x0)))
                y0 = max(0.0, min(1.0, float(y0)))
                x1 = max(0.0, min(1.0, float(x1)))
                y1 = max(0.0, min(1.0, float(y1)))
                left = max(0, min(width - 1, int(round(width * min(x0, x1)))))
                top = max(0, min(height - 1, int(round(height * min(y0, y1)))))
                right = max(left + 1, min(width, int(round(width * max(x0, x1)))))
                bottom = max(top + 1, min(height, int(round(height * max(y0, y1)))))
            else:
                top = int(height * max(0.0, min(float(crop_top), 0.3)))
                bottom = height - int(height * max(0.0, min(float(crop_bottom), 0.3)))
                left, right = 0, width
                bottom = max(top + 1, min(height, bottom))
            box = (left, top, right, bottom)
            masked = Image.new("RGB", image.size, _estimate_mask_background(image))
            region = image.crop(box)
            try:
                masked.paste(region, (left, top))
                # Write to a sibling temporary file and atomically replace the
                # final cache entry.  Interrupted OCR can never leave a partial
                # PNG that a later model mistakes for a valid shared crop.
                tmp = dest.with_name(dest.name + f".tmp-{threading.get_ident()}")
                try:
                    image_format = (dest.suffix.lstrip(".") or "PNG").upper()
                    if image_format == "JPG":
                        image_format = "JPEG"
                    masked.save(tmp, format=image_format)
                    tmp.replace(dest)
                finally:
                    tmp.unlink(missing_ok=True)
            finally:
                region.close()
                masked.close()
        finally:
            image.close()

    print(
        f"  🎭  正文区域掩膜: {src.name} 保持 {width}x{height} → 保留区域 {box} "
        f"→ 框外替换为纸白色 → {dest}"
    )
    return str(dest)


# OCR 识别本身（快捷指令）已经拆到
# adapters/vision_backends/ 包里，走 BackendFactory 统一调用——新增一种
# 识别方式只需要在那边加一个 backend，这里完全不用改。
from adapters.vision_backends import BackendFactory, OCRConfig

# ── 页眉/页脚检测（沿用原脚本算法，稍作增强）────────────────────────────────

def detect_running_headers(all_lines_per_page: list[list[str]]) -> set[str]:
    """
    统计每行在多少张不同图片中出现。
    出现比例 > HEADER_THRESHOLD_RATIO 且长度 ≤ HEADER_MAX_LEN → 视为页眉/页脚。
    """
    n = len(all_lines_per_page)
    counter: Counter = Counter()
    for lines in all_lines_per_page:
        for line in set(lines):
            stripped = line.strip()
            if stripped:
                counter[stripped] += 1

    threshold = max(2, int(n * HEADER_THRESHOLD_RATIO) + 1)
    return {
        line for line, cnt in counter.items()
        if cnt >= threshold and len(line) <= HEADER_MAX_LEN
    }


def filter_lines(lines: list[str], running_headers: set[str]) -> list[str]:
    """过滤页码行、页眉/页脚行"""
    kept = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if PAGE_NUMBER_RE.match(s):
            continue
        if s in running_headers:
            continue
        kept.append(s)
    return kept


# ── 页面自动分类 ──────────────────────────────────────────────────────────────

def _chapter_title_lines(lines: list[str]) -> list[str]:
    """Return chapter/title lines that must survive low-text page filtering."""
    return [str(line).strip() for line in lines if CHAPTER_RE.match(str(line).strip())]

def classify_page(image_path: str, filtered_lines: list[str]) -> tuple[BlockType, float]:
    """
    根据过滤后的文字内容，自动判定页面类型。
    返回 (BlockType, confidence)。

    分类逻辑（优先级从高到低）：
        1. 空白页：过滤后字数 < BLANK_THRESHOLD
        2. 封面：第一页 + 字数很少（由调用方额外判断）
        3. 插图：字数 < MIN_CONTENT_CHARS；或字数刚过线但标点数 < MIN_PUNCT_COUNT
        4. 正文：其余

    标点数检查只在字数刚过线（< MIN_CONTENT_CHARS*3）时才生效，用来兜底"字数
    勉强达标但内容像是杂散文字/标题"的边界情况——字数已经很充分时不再靠标点
    数二次否决，因为有些 OCR 引擎（比如 PaddleOCR）在稠密竖排文本上偶尔会漏识别
    句读符号，明明识别出了几百字正文却因为标点太少被误判成插图/封面。
    """
    joined = "".join(filtered_lines)
    char_count = len(joined)
    punct_count = sum(1 for ch in joined if ch in PUNCT_CHARS)

    # A chapter opener is often a single short line with little punctuation.
    # It is text, not an illustration, and must remain available to the normal
    # title/section classifier instead of being discarded as IMAGE_REF.
    if _chapter_title_lines(filtered_lines):
        return BlockType.PARAGRAPH, 0.98

    if char_count < BLANK_THRESHOLD:
        return BlockType.BLANK, 0.95

    if char_count < MIN_CONTENT_CHARS:
        return BlockType.ILLUSTRATION, 0.80

    if char_count < MIN_CONTENT_CHARS * 3 and punct_count < MIN_PUNCT_COUNT:
        return BlockType.ILLUSTRATION, 0.80

    return BlockType.PARAGRAPH, 0.90   # 暂用 PARAGRAPH 表示"正文页"


def auto_classify_pages(
    image_paths: list[str],
    all_filtered_lines: list[list[str]],
) -> list[BlockType]:
    """
    对所有页面做自动分类：
      - 第 1 页且字符少 → cover
      - 之后的逻辑同 classify_page
    """
    result = []
    for i, (path, lines) in enumerate(zip(image_paths, all_filtered_lines)):
        ptype, _ = classify_page(path, lines)
        # 第一页字数少的一定是封面
        if i == 0 and ptype in (BlockType.BLANK, BlockType.ILLUSTRATION):
            ptype = BlockType.COVER
        result.append(ptype)
    return result


# ── 正文块拆分 ────────────────────────────────────────────────────────────────

def lines_to_blocks(
    lines: list[str], page_no: int, start_order: int, ocr_mode: str = "ja_vertical"
) -> list[Block]:
    """
    把过滤后的文字行转换为 Block 列表。
    - 匹配章节正则 → BlockType.CHAPTER
    - 以「/『开头结尾 → BlockType.DIALOGUE
    - 其余 → BlockType.PARAGRAPH
    """
    blocks: list[Block] = []
    order = start_order

    for line in lines:
        s = line.strip()
        if not s:
            continue

        if is_chapter_title(s, ocr_mode):
            btype = BlockType.CHAPTER
        elif s.startswith(('「', '『', '“', '‘')) or s.endswith(('」', '』', '”', '’')):
            btype = BlockType.DIALOGUE
        else:
            btype = BlockType.PARAGRAPH

        blocks.append(Block(
            type=btype,
            text=s,
            ocr_raw=s,         # 保留原始，供 Formatter diff 使用
            page=page_no,
            reading_order=order,
            confidence=0.95,   # Apple Vision 典型置信度
            text_direction=("horizontal" if is_chinese_horizontal(ocr_mode) else "vertical"),
            source_format="ocr",
        ))
        order += 1

    return blocks


# ── 主入口 ────────────────────────────────────────────────────────────────────

def run(
    image_folder: str | None = None,
    page_overrides: dict[int, str] | None = None,
    shortcut_name: str = SHORTCUT_NAME,
    verbose: bool = True,
    input_paths: list[str] | None = None,
    progress_callback=None,
    cancel_check=None,
    crop_top: float = 0.0,
    crop_bottom: float = 0.0,
    crop_rect: tuple[float, float, float, float] | None = None,
    backend: str = "auto",
    vertical: bool = True,
    recognition_level: str = "accurate",
    recognition_languages: list[str] | None = None,
    use_language_correction: bool = True,
    automatically_detect_language: bool = False,
    minimum_text_height_fraction: float = 0.005,
    candidate_count: int = 3,
    orientation: str = "auto",
    vertical_preprocess: str = "none",
    temp_crop_dir: str | None = None,
    reuse_existing_crops: bool = False,
    filter_running_headers: bool = True,
    ocr_mode: str = "ja_vertical",
    merge_horizontal_fragments: bool = True,
) -> UnifiedDocument:
    """
    核心函数：对输入执行 OCR，返回 UnifiedDocument。

    Args:
        image_folder:   图片文件夹路径（向后兼容，等价于 input_paths=[image_folder]）
        input_paths:    混合输入列表，支持文件夹 / PDF / 单张或多张图片路径混合传入。
                        例如 ["/imgs/"], ["a.pdf"], ["p1.jpg","p2.png","b.pdf"]
                        优先于 image_folder。
        page_overrides: {页码(1-based): BlockType字符串} —— Page Manager 手动标注覆盖自动分类
        shortcut_name:  快捷指令名称（backend="shortcut" 时生效）
        verbose:        是否打印进度
        progress_callback: 可选回调 (current: int, total: int, filename: str, image_path: str) -> None，
                        每处理完一页调用一次，供 GUI 更新真实进度条 / 实时预览当前图片。
        cancel_check:   可选回调 () -> bool，每页处理前调用；返回 True 则立即中止，
                        返回已处理页面的部分结果（供 GUI 实现"暂停/取消"）。
        crop_top:       识别前裁掉页面顶部的比例（0.0~0.3），把页眉整个排除在
                        OCR 识别区域之外，从源头上不识别页眉文字。
        crop_bottom:    同上，裁掉底部（页脚区域）。
        crop_rect:      手动框选的识别区域 (x0,y0,x1,y1)，归一化坐标，优先于
                        crop_top/crop_bottom（GUI 拖框选工具产生的参数）。
        backend:        "live_text" 使用 VisionKit ImageAnalyzer；"native_helper" 使用 RecognizeTextRequest；"shortcut" 保留原快捷指令；"auto" 优先 Live Text、基础设施/可用性失败时回退快捷指令。
        vertical:       竖排阅读顺序提示；Helper 会按右到左、上到下排序观察结果。
        vertical_preprocess: "crop_rotate_left" 时仅对单个狭长竖列紧裁并左旋 90°后识别一次。
        filter_running_headers: 是否在组装文档前自动删除跨页重复页眉/页脚。
                        启用“逐列成句”时应传 False，避免误删页面边缘正文列。

    Returns:
        UnifiedDocument
    """
    global SHORTCUT_NAME
    ocr_mode = normalize_ocr_mode(ocr_mode)
    profile = get_ocr_profile(ocr_mode)
    # Chinese horizontal mode is a hard profile boundary: no vertical sorting,
    # rotation or Japanese language list can leak in from a previous run.
    vertical = bool(vertical and profile.vertical)
    if is_chinese_horizontal(ocr_mode):
        recognition_languages = list(profile.apple_languages)
        vertical_preprocess = "none"
    elif not recognition_languages:
        recognition_languages = list(profile.apple_languages)
    if backend in {None, ""}:
        backend = "auto"
    if backend not in {"auto", "live_text", "native_helper", "shortcut"}:
        raise ValueError(f"不支持的 Apple Vision backend: {backend}")

    SHORTCUT_NAME = shortcut_name

    overrides = {int(k): v for k, v in (page_overrides or {}).items()}

    # ── 收集图片路径（支持文件夹 / PDF / 单图混合）───────────────────────────
    from adapters.pdf_input import expand_inputs, natural_sort_key

    raw_inputs = input_paths if input_paths else ([image_folder] if image_folder else [])
    if not raw_inputs:
        raise ValueError("必须提供 image_folder 或 input_paths")

    work_dir = image_folder or str(Path(raw_inputs[0]).parent)
    image_paths = expand_inputs(
        raw_inputs, work_dir=work_dir, cancel_check=cancel_check
    )
    image_paths = sorted(set(image_paths), key=natural_sort_key)

    if not image_paths:
        raise FileNotFoundError(f"未找到可处理的图片/PDF: {raw_inputs}")

    if verbose:
        print(f"📂  共 {len(image_paths)} 张图片（含 PDF 转换页面）")

    # 只有明确标注为"正文"的页面才会实际跑 OCR；封面/扉页/目录/插图/版权页/
    # 后记……其它任何标注类型一律跳过（省时间，也避免杂散文字被误当成页眉
    # 参与跨页统计）。完全没有标注过的页面（既没有手动覆盖也没有走过 Page
    # Manager 自动分类）仍然照旧 OCR 一遍，靠识别出的文字量事后自动判断类型，
    # 作为兜底。进度回调（含实时预览）只对"会实际 OCR 的页面"触发，分母也
    # 只算这些页——封面/插图等页不会出现在 GUI 的识别进度/预览里。
    def _skip_ocr(page_no: int) -> bool:
        return should_skip_page_ocr(overrides.get(page_no))

    text_page_total = sum(1 for i in range(1, len(image_paths) + 1) if not _skip_ocr(i))

    ocr_backend = BackendFactory.create(backend, vertical=vertical)
    available, reason = ocr_backend.is_available()
    if not available:
        raise RuntimeError(f"OCR backend {backend!r} 不可用：{reason}")
    ocr_config = OCRConfig(
        shortcut_name=shortcut_name,
        vertical=vertical,
        recognition_level=recognition_level,
        languages=list(recognition_languages or profile.apple_languages),
        use_language_correction=use_language_correction,
        automatically_detect_language=automatically_detect_language,
        minimum_text_height_fraction=minimum_text_height_fraction,
        candidate_count=candidate_count,
        orientation=orientation,
        vertical_preprocess=vertical_preprocess,
    )

    # ── 第一遍：OCR 所有页面 ─────────────────────────────────────────────────
    raw_lines_per_page: list[list[str]] = []
    cancelled = False
    ocr_progress = 0
    for i, path in enumerate(image_paths, 1):
        if cancel_check is not None and cancel_check():
            cancelled = True
            if verbose:
                print(f"\n⏸  已在第 {i}/{len(image_paths)} 页暂停，保留已识别的 {i - 1} 页")
            break

        skip_ocr = _skip_ocr(i)

        if skip_ocr:
            if verbose:
                print(f"  [{i:3d}/{len(image_paths)}] 跳过 OCR（已标注为 {overrides.get(i)}）: {os.path.basename(path)}")
            raw_lines_per_page.append([])
        else:
            if verbose:
                print(f"  [{i:3d}/{len(image_paths)}] OCR: {os.path.basename(path)}", end=" ")
            ocr_path = crop_for_ocr(
                path,
                crop_top=crop_top,
                crop_bottom=crop_bottom,
                crop_rect=crop_rect,
                out_dir=temp_crop_dir,
                reuse_existing=bool(reuse_existing_crops),
            )
            # page 由调用方回填，不是 backend 自己知道的——backend.recognize()
            # 只认识"这一张图"，不知道自己在整本书里是第几页。
            ocr_result = ocr_backend.recognize(ocr_path, ocr_config)
            ocr_result.page = i
            raw = ocr_result.full_text
            lines = raw.splitlines()
            raw_lines_per_page.append(lines)
            if verbose:
                print(f"→ {len(raw)} 字")
            ocr_progress += 1
            if progress_callback is not None:
                progress_callback(ocr_progress, text_page_total, os.path.basename(path), path)

    ocr_backend.close()

    if cancelled:
        # 只保留已经识别完成的页面，其余截断，避免下游按索引访问越界
        image_paths = image_paths[:len(raw_lines_per_page)]

    # ── 可选页眉/页脚过滤 ────────────────────────────────────────────────────
    # 逐列成句开启时不允许在重建前凭文本频率删除任何列；否则页面边缘的正文
    # 一旦误判为页眉，跨页接续阶段无法恢复。此模式只去除空行，页眉清理由
    # Formatter 的可选“清理模块”承担。
    running_headers = (
        detect_running_headers(raw_lines_per_page) if filter_running_headers else set()
    )
    if verbose and running_headers:
        print(f"\n🔍  检测到页眉/页脚（将过滤）：")
        for h in sorted(running_headers):
            print(f"    「{h}」")
    elif verbose and not filter_running_headers:
        print("\n🛡️  逐列成句保护：跳过自动页眉/页脚删除，保留 OCR 原始文字列")

    if filter_running_headers:
        filtered_lines_per_page = [
            normalize_lines(filter_lines(lines, running_headers), ocr_mode)
            for lines in raw_lines_per_page
        ]
    else:
        filtered_lines_per_page = [
            normalize_lines(lines, ocr_mode)
            for lines in raw_lines_per_page
        ]

    # ── 自动页面分类 ─────────────────────────────────────────────────────────
    auto_types = (
        auto_classify_pages_profile(image_paths, filtered_lines_per_page, ocr_mode)
        if is_chinese_horizontal(ocr_mode)
        else auto_classify_pages(image_paths, filtered_lines_per_page)
    )

    # ── 构建 UnifiedDocument ─────────────────────────────────────────────────
    doc = UnifiedDocument()
    doc.metadata = Metadata(
        source_engine="apple_vision",
        language=profile.language,
    )
    apply_profile_metadata(doc, ocr_mode)

    order_counter = 0
    skipped_image_count = 0
    text_page_count = 0
    chapter_index = 0

    for i, (path, filtered_lines) in enumerate(zip(image_paths, filtered_lines_per_page)):
        page_no = i + 1
        fname = os.path.basename(path)

        # 应用手动覆盖（Page Manager 标注优先）
        if page_no in overrides:
            try:
                ptype = BlockType(overrides[page_no])
                conf = 1.0
            except ValueError:
                ptype = auto_types[i]
                conf = 0.90
        else:
            ptype = auto_types[i]
            conf = 0.90

        # 记录页面元信息
        doc.pages.append(PageInfo(
            page_no=page_no,
            page_type=ptype,
            image_path=path,
            confidence=conf,
        ))

        # ── 根据页面类型决定处理方式 ─────────────────────────────────────────
        # 只有"正文"页会生成文字块；空白页整页跳过；其余任何类型
        # （封面/扉页/目录/插图/版权页/后记……）一律按原图顺序放进
        # IMAGE_REF，直接按页面顺序进入 EPUB。
        if ptype == BlockType.BLANK:
            # 空白页：不产生任何 block
            if verbose:
                print(f"  ·空白  第{page_no}页: {fname}")

        elif ptype != BlockType.PARAGRAPH:
            # 非正文页：插入 IMAGE_REF block，保留原图路径
            # 锚点指向上一个 block（插图/其它非正文页按顺序排列用）
            last_idx = len(doc.blocks) - 1
            anchor = f"block_{last_idx}" if last_idx >= 0 else "start"

            doc.blocks.append(Block(
                type=BlockType.IMAGE_REF,
                image_path=path,
                image_anchor=anchor,
                page=page_no,
                reading_order=order_counter,
                confidence=conf,
            ))
            order_counter += 1
            skipped_image_count += 1
            if verbose:
                print(f"  🖼️  图片  第{page_no}页: {fname} ({ptype.value})  anchor={anchor}")

        else:
            # 正文页（PARAGRAPH 类型 = 自动分类为正文）
            new_blocks = lines_to_blocks(
                filtered_lines, page_no, order_counter, ocr_mode=ocr_mode
            )

            for b in new_blocks:
                # 更新章节计数
                if b.type == BlockType.CHAPTER:
                    chapter_index += 1
                    b.chapter_index = chapter_index

                    from models.document import TocEntry
                    doc.toc.append(TocEntry(
                        title=b.text,
                        chapter_index=chapter_index,
                        block_index=len(doc.blocks),
                    ))

                doc.blocks.append(b)

            order_counter += len(new_blocks)
            text_page_count += 1
            if verbose:
                print(f"  📄 正文  第{page_no}页: {fname}  → {len(new_blocks)} 块")

    # ── 写入处理日志 ─────────────────────────────────────────────────────────
    doc.add_log(
        "ocr_profile",
        f"OCR 模式：{profile.label} · {profile.language} · {profile.writing_direction}",
        len(image_paths),
    )
    if cancelled:
        doc.add_log("apple_vision_adapter", f"OCR 已暂停，仅识别 {len(image_paths)} 页", len(image_paths))
    else:
        doc.add_log("apple_vision_adapter", "OCR完成", len(image_paths))
    if filter_running_headers:
        doc.add_log("header_filter", f"过滤页眉: {sorted(running_headers)}", len(running_headers))
    else:
        doc.add_log("header_filter", "逐列成句保护：未执行自动页眉/页脚删除", 0)
    doc.add_log("page_classify", f"正文页{text_page_count}，图片页{skipped_image_count}，空白页{len(image_paths)-text_page_count-skipped_image_count}")
    skipped_before_ocr = sum(1 for page_no in range(1, len(image_paths) + 1) if _skip_ocr(page_no))
    if skipped_before_ocr:
        doc.add_log(
            "ocr_page_admission",
            f"OCR 前置隔离 {skipped_before_ocr} 个已分类非正文页；未调用 Apple Vision",
            skipped_before_ocr,
        )
    doc.add_log("chapter_detect", f"识别章节 {chapter_index} 个", chapter_index)

    if verbose:
        print(f"\n✅  完成: {text_page_count} 正文页，{skipped_image_count} 图片页，{chapter_index} 章节，{len(doc.blocks)} 个块")

    return doc


# ── CLI 入口 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apple Vision OCR → 统一文档模型 JSON"
    )
    parser.add_argument("image_folder", help="图片文件夹路径")
    parser.add_argument("output_json", help="输出 JSON 路径")
    parser.add_argument(
        "--overrides", "-o",
        help="Page Manager 手动标注文件（JSON）：{\"1\":\"cover\",\"5\":\"blank\",...}",
        default=None,
    )
    parser.add_argument(
        "--shortcut", "-s",
        help=f"快捷指令名称（默认: {SHORTCUT_NAME}）",
        default=SHORTCUT_NAME,
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["auto", "live_text", "native_helper", "shortcut"],
        default="auto",
        help="OCR backend：auto / live_text / native_helper / shortcut",
    )
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.overrides:
        with open(args.overrides, encoding="utf-8") as f:
            overrides = json.load(f)

    doc = run(
        image_folder=args.image_folder,
        page_overrides=overrides,
        shortcut_name=args.shortcut,
        backend=args.backend,
        vertical=True,
        verbose=not args.quiet,
    )

    with open(args.output_json, "w", encoding="utf-8") as f:
        f.write(doc.to_json())

    print(f"\n💾  已写入: {args.output_json}")


if __name__ == "__main__":
    main()
