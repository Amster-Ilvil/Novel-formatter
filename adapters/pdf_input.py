#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF → 图片 转换工具
用于让 Apple Vision 适配器也能处理 PDF 输入（PDF 页面逐页转成 PNG 后走同一条 OCR 流水线）。

依赖（任选其一，优先级从高到低）：
    pip install pymupdf     # 推荐：快、无需系统依赖
    pip install pdf2image + poppler (brew install poppler)   # 备选
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

_NUM_RE = re.compile(r'(\d+)')


def natural_sort_key(path) -> list:
    """
    "自然排序"键：把文件名拆成数字/非数字片段，数字部分按数值比较。
    普通字符串排序会把 page_2.png 排在 page_10.png 后面（'1'<'2' 字典序），
    这里按数值比较，得到符合直觉、和 Finder 一致的 1,2,...,10,11 顺序。
    """
    name = Path(path).name
    return [int(tok) if tok.isdigit() else tok.lower() for tok in _NUM_RE.split(name)]


def pdf_available() -> tuple[bool, str]:
    """检测可用的 PDF 转换后端"""
    try:
        import fitz  # PyMuPDF
        return True, "pymupdf"
    except ImportError:
        pass
    try:
        import pdf2image
        return True, "pdf2image"
    except ImportError:
        pass
    return False, ""


def pdf_to_images(pdf_path: str, out_dir: str, dpi: int = 200) -> list[str]:
    """
    把 PDF 每一页渲染成 PNG，保存到 out_dir，返回图片路径列表（按页码排序）。
    文件名格式: {pdf_stem}_p{page:04d}.png
    """
    ok, backend = pdf_available()
    if not ok:
        raise RuntimeError(
            "未安装 PDF 转换库。请运行以下任一命令后重试：\n"
            "  pip3 install pymupdf\n"
            "  或\n"
            "  pip3 install pdf2image  &&  brew install poppler"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf_path).stem
    result: list[str] = []

    if backend == "pymupdf":
        import fitz
        doc = fitz.open(pdf_path)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat)
            fname = out / f"{stem}_p{i+1:04d}.png"
            pix.save(str(fname))
            result.append(str(fname))
        doc.close()

    else:  # pdf2image
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path, dpi=dpi)
        for i, page in enumerate(pages):
            fname = out / f"{stem}_p{i+1:04d}.png"
            page.save(str(fname), "PNG")
            result.append(str(fname))

    return result


def expand_inputs(paths: list[str], work_dir: str) -> list[str]:
    """
    接收混合输入列表（图片路径 / PDF路径 / 文件夹路径），
    展开为统一的图片路径列表：
        - 文件夹 → 内部所有图片（按文件名排序）
        - PDF    → 转换后的每页 PNG
        - 图片   → 原样保留
    """
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.heic', '.tif', '.tiff', '.bmp', '.gif'}
    images: list[str] = []

    for p in paths:
        path = Path(p)
        if path.is_dir():
            sub = sorted(
                [str(f) for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTS],
                key=natural_sort_key
            )
            images.extend(sub)
        elif path.suffix.lower() == '.pdf':
            pdf_out = Path(work_dir) / f"_pdf_{path.stem}"
            converted = pdf_to_images(str(path), str(pdf_out))
            images.extend(converted)
        elif path.suffix.lower() in IMAGE_EXTS:
            images.append(str(path))

    return images
