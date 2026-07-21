#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleOCR 适配器（macOS，跑在独立 venv 里）

背景：
    PaddlePaddle 目前没有 Python 3.14 的预编译包，而本项目其余部分跑在系统
    默认的 3.14 上。所以这里不直接 import paddleocr，而是通过 subprocess
    调用 .venv-paddle（Python 3.13，见 adapters/paddle_ocr_worker.py）里的
    解释器，跨进程拿到逐行识别结果（含像素坐标），再在这一侧（有 PIL）
    转换成归一化 bbox，组装成 UnifiedDocument。

    页眉检测 / 页面自动分类 / 章节正则复用 apple_vision_adapter 里已有的实现，
    避免重复一份几乎一样的规则。PaddleOCR 的检测顺序不是可靠的阅读顺序
    （尤其竖排日文），所以这里给每个 Block 都写入 bbox——真正的阅读顺序
    由 Formatter 里的 reading_order 步骤（engine/reading_order.py）用坐标
    重新算，这里不需要、也不应该自己排。

依赖：
    .venv-paddle/（用 Python 3.13 创建，装了 paddlepaddle + paddleocr）
    首次运行会创建：
        /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m venv .venv-paddle
        .venv-paddle/bin/pip install paddlepaddle paddleocr

用法（命令行，供单独测试）：
    python adapters/paddle_ocr_adapter.py /图片文件夹 output.json
"""

from __future__ import annotations

import sys
import os
import json
import subprocess
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import (
    UnifiedDocument, Block, BlockType, PageInfo, BoundingBox, Metadata, TocEntry
)
from adapters.apple_vision_adapter import (
    detect_running_headers, auto_classify_pages, CHAPTER_RE,
)

VENV_DIR = Path(__file__).parent.parent / ".venv-paddle"
VENV_PYTHON = VENV_DIR / "bin" / "python"
WORKER_SCRIPT = Path(__file__).parent / "paddle_ocr_worker.py"


def _venv_ready() -> bool:
    return VENV_PYTHON.exists()


def _has_structure_deps() -> bool:
    """paddlex[ocr] 额外依赖（PP-StructureV3 / PaddleOCRVL 都需要）是否已装"""
    result = subprocess.run(
        [str(VENV_PYTHON), "-c", "import openpyxl"],
        capture_output=True,
    )
    return result.returncode == 0


def setup_venv(verbose: bool = True, pipeline: str = "ocr") -> None:
    """创建 .venv-paddle 并安装 paddlepaddle + paddleocr（幂等，已存在则跳过）"""
    if not _venv_ready():
        py313_candidates = [
            "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3",
            "/opt/homebrew/bin/python3.13",
            "/usr/local/bin/python3.13",
        ]
        py313 = next((p for p in py313_candidates if Path(p).exists()), None)
        if py313 is None:
            raise RuntimeError(
                "找不到 Python 3.13（PaddlePaddle 暂不支持 Python 3.14）。"
                "请先安装 Python 3.13，例如 `brew install python@3.13`。"
            )

        if verbose:
            print(f"🔧  首次使用 PaddleOCR：用 {py313} 创建独立虚拟环境 {VENV_DIR} ...")
        subprocess.run([py313, "-m", "venv", str(VENV_DIR)], check=True)
        subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip", "--quiet"], check=True)
        if verbose:
            print("📦  安装 paddlepaddle + paddleocr（首次约 1~1.5GB，需要几分钟）...")
        subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "paddlepaddle", "paddleocr"], check=True)

    # PP-StructureV3 / PaddleOCRVL 都需要 paddlex[ocr] 这组额外依赖（版面分析、
    # 表格/公式解析用到的库），基础安装不含，第一次选到这两个 pipeline 时才装。
    if pipeline in ("structure", "vl") and not _has_structure_deps():
        if verbose:
            print(f"📦  {pipeline} 模型需要额外依赖包，首次使用需要安装（约几百 MB）...")
        subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "paddlex[ocr]"], check=True)


def _page_size(image_path: str) -> tuple[int, int]:
    from PIL import Image
    with Image.open(image_path) as img:
        return img.size  # (w, h)


def _run_worker(image_paths: list[str], lang: str, pipeline: str = "ocr", cancel_check=None):
    """
    以子进程方式运行 paddle_ocr_worker.py，按行 yield (path, blocks|None, error|None)。
    blocks: [{"text":..., "confidence":..., "box": [[x,y]x4]}, ...]
    pipeline: "ocr"（默认，纯文字识别）/ "structure"（PP-StructureV3，版面分析）/
              "vl"（PaddleOCR-VL，视觉语言模型文档解析，首次用需下载大模型）
    """
    cmd = [str(VENV_PYTHON), str(WORKER_SCRIPT), "--lang", lang, "--pipeline", pipeline, *image_paths]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    try:
        for line in proc.stdout:
            if cancel_check is not None and cancel_check():
                proc.terminate()
                break
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("ok"):
                yield data["path"], data["blocks"], None
            else:
                yield data["path"], None, data.get("error", "未知错误")
    finally:
        proc.stdout.close()
        stderr_tail = proc.stderr.read()
        proc.stderr.close()
        ret = proc.wait()
        if ret != 0 and ret != -15:  # -15 = terminated（取消）
            raise RuntimeError(f"PaddleOCR worker 异常退出 (code={ret}):\n{stderr_tail[-2000:]}")


def run(
    image_folder: str | None = None,
    page_overrides: dict[int, str] | None = None,
    lang: str = "japan",
    pipeline: str = "ocr",
    verbose: bool = True,
    input_paths: list[str] | None = None,
    progress_callback=None,
    cancel_check=None,
    crop_top: float = 0.0,
    crop_bottom: float = 0.0,
    crop_rect: tuple[float, float, float, float] | None = None,
    temp_crop_dir: str | None = None,
) -> UnifiedDocument:
    """
    核心函数：对输入执行 PaddleOCR，返回 UnifiedDocument。
    参数与 apple_vision_adapter.run() 保持一致（含 crop_rect 手动框选区域），
    方便 GUI 按适配器名切换调用。

    pipeline: "ocr"（默认，PP-OCR 纯文字识别，最快）/
              "structure"（PP-StructureV3，多一步文档方向矫正/版面分析，
              需要 pip install "paddlex[ocr]"）/
              "vl"（PaddleOCR-VL，视觉语言模型，识别质量通常更好但模型体积
              是前两者的好几倍，首次使用会下载数 GB 模型文件）。
    """
    setup_venv(verbose=verbose, pipeline=pipeline)

    overrides = {int(k): v for k, v in (page_overrides or {}).items()}

    from adapters.pdf_input import expand_inputs, natural_sort_key

    raw_inputs = input_paths if input_paths else ([image_folder] if image_folder else [])
    if not raw_inputs:
        raise ValueError("必须提供 image_folder 或 input_paths")

    work_dir = image_folder or str(Path(raw_inputs[0]).parent)
    image_paths = expand_inputs(raw_inputs, work_dir=work_dir)
    image_paths = sorted(set(image_paths), key=natural_sort_key)

    if not image_paths:
        raise FileNotFoundError(f"未找到可处理的图片/PDF: {raw_inputs}")

    if crop_rect is not None or crop_top > 0 or crop_bottom > 0:
        from adapters.apple_vision_adapter import crop_for_ocr
        ocr_inputs = [
            crop_for_ocr(p, crop_top=crop_top, crop_bottom=crop_bottom, crop_rect=crop_rect, out_dir=temp_crop_dir)
            for p in image_paths
        ]
    else:
        ocr_inputs = image_paths

    if verbose:
        print(f"📂  共 {len(image_paths)} 张图片（含 PDF 转换页面）")

    # 只 OCR 未被手动标注为"非正文"的页面
    pages_to_ocr = []
    for i, path in enumerate(image_paths, 1):
        override_type = overrides.get(i)
        skip_ocr = bool(override_type) and override_type != BlockType.PARAGRAPH.value
        if not skip_ocr:
            pages_to_ocr.append((i, ocr_inputs[i - 1]))

    # ── OCR：逐页拿到 [{"text","confidence","box"}] ─────────────────────────
    raw_items_per_page: list[list[dict]] = [[] for _ in image_paths]
    ocr_results = {}
    if pages_to_ocr:
        ocr_paths = [p for _, p in pages_to_ocr]
        for path, blocks, error in _run_worker(ocr_paths, lang=lang, pipeline=pipeline, cancel_check=cancel_check):
            ocr_results[path] = (blocks, error)

    cancelled = False
    processed = 0
    for i, path in pages_to_ocr:
        if cancel_check is not None and cancel_check() and path not in ocr_results:
            cancelled = True
            break
        blocks, error = ocr_results.get(path, (None, "未识别（可能已取消）"))
        if error:
            if verbose:
                print(f"  ⚠️  第{i}页识别失败: {error}")
            blocks = []
        raw_items_per_page[i - 1] = blocks
        processed += 1
        if verbose:
            print(f"  [{i:3d}/{len(image_paths)}] OCR → {len(blocks)} 个文本块")
        if progress_callback is not None:
            # 预览用原图（未裁剪），这样 GUI 上画的框选矩形才对得上显示的图片
            progress_callback(processed, len(pages_to_ocr), os.path.basename(path), image_paths[i - 1])

    # ── 页眉/页脚检测与过滤（复用 apple_vision_adapter 的规则）────────────────
    all_texts_per_page = [[it["text"] for it in items] for items in raw_items_per_page]
    running_headers = detect_running_headers(all_texts_per_page)
    if verbose and running_headers:
        print(f"\n🔍  检测到页眉/页脚（将过滤）：")
        for h in sorted(running_headers):
            print(f"    「{h}」")

    filtered_items_per_page = [
        [it for it in items if it["text"].strip() not in running_headers]
        for items in raw_items_per_page
    ]

    filtered_texts_per_page = [[it["text"] for it in items] for items in filtered_items_per_page]
    auto_types = auto_classify_pages(image_paths, filtered_texts_per_page)

    # ── 构建 UnifiedDocument ─────────────────────────────────────────────────
    doc = UnifiedDocument()
    doc.metadata = Metadata(source_engine="paddle_ocr", language="ja")

    order_counter = 0
    skipped_image_count = 0
    text_page_count = 0
    chapter_index = 0

    for i, path in enumerate(image_paths):
        page_no = i + 1
        fname = os.path.basename(path)

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

        doc.pages.append(PageInfo(page_no=page_no, page_type=ptype, image_path=path, confidence=conf))

        if ptype == BlockType.BLANK:
            if verbose:
                print(f"  ·空白  第{page_no}页: {fname}")
            continue

        if ptype != BlockType.PARAGRAPH:
            last_idx = len(doc.blocks) - 1
            anchor = f"block_{last_idx}" if last_idx >= 0 else "start"
            doc.blocks.append(Block(
                type=BlockType.IMAGE_REF, image_path=path, image_anchor=anchor,
                page=page_no, reading_order=order_counter, confidence=conf,
            ))
            order_counter += 1
            skipped_image_count += 1
            if verbose:
                print(f"  🖼️  图片  第{page_no}页: {fname} ({ptype.value})  anchor={anchor}")
            continue

        # 正文页：需要页面像素尺寸才能把 box 归一化成 bbox——注意要用实际送
        # 去 OCR 的那张图（裁剪过的话是裁剪后的），而不是原图，否则裁剪开启
        # 时坐标系对不上，bbox 会算错。
        try:
            page_w, page_h = _page_size(ocr_inputs[i])
        except Exception:
            page_w, page_h = (0, 0)

        for item in filtered_items_per_page[i]:
            text = item["text"].strip()
            if not text:
                continue

            if CHAPTER_RE.match(text):
                btype = BlockType.CHAPTER
            elif text.startswith(('「', '『')) or text.endswith(('」', '』')):
                btype = BlockType.DIALOGUE
            else:
                btype = BlockType.PARAGRAPH

            bbox = None
            box = item.get("box")
            if box and page_w and page_h:
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                bbox = BoundingBox.from_pixels(min(xs), min(ys), max(xs), max(ys), page_w, page_h)

            block = Block(
                type=btype, text=text, ocr_raw=text, page=page_no,
                bbox=bbox, reading_order=order_counter,
                confidence=item.get("confidence", 0.9),
            )

            if btype == BlockType.CHAPTER:
                chapter_index += 1
                block.chapter_index = chapter_index
                doc.toc.append(TocEntry(title=text, chapter_index=chapter_index, block_index=len(doc.blocks)))

            doc.blocks.append(block)
            order_counter += 1

        text_page_count += 1
        if verbose:
            print(f"  📄 正文  第{page_no}页: {fname}  → {len(filtered_items_per_page[i])} 块")

    if cancelled:
        doc.add_log("paddle_ocr_adapter", f"OCR 已暂停，仅识别 {processed}/{len(pages_to_ocr)} 页", processed)
    else:
        doc.add_log("paddle_ocr_adapter", "OCR完成", len(image_paths))
    doc.add_log("header_filter", f"过滤页眉: {sorted(running_headers)}", len(running_headers))
    doc.add_log("page_classify", f"正文页{text_page_count}，图片页{skipped_image_count}，空白页{len(image_paths)-text_page_count-skipped_image_count}")
    doc.add_log("chapter_detect", f"识别章节 {chapter_index} 个", chapter_index)

    if verbose:
        print(f"\n✅  完成: {text_page_count} 正文页，{skipped_image_count} 图片页，{chapter_index} 章节，{len(doc.blocks)} 个块")

    return doc


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR 适配器 CLI")
    parser.add_argument("input", help="图片文件夹或文件路径")
    parser.add_argument("output", help="输出 JSON 路径")
    parser.add_argument("--lang", default="japan")
    args = parser.parse_args()

    doc = run(image_folder=args.input, lang=args.lang)
    Path(args.output).write_text(doc.to_json(), encoding="utf-8")
    print(f"已写入 {args.output}")


if __name__ == "__main__":
    main()
