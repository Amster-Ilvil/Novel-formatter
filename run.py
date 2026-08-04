#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Novel Formatter Studio — 一键运行脚本

完整流程：
    图片文件夹
        → Apple Vision OCR（via macOS 快捷指令）
        → 统一文档模型 JSON（可选保存中间结果）
        → Novel Formatter Engine（六步处理）
        → EPUB3 输出

用法：
    # 最简：图片文件夹 → EPUB
    python run.py /path/to/images output.epub

    # 指定书名作者
    python run.py /path/to/images output.epub --title "魔法科高校の劣等生" --author "佐島勤"

    # 使用 Page Manager 手动标注（JSON格式）
    python run.py /path/to/images output.epub --overrides overrides.json

    # 保存中间 JSON（方便调试或跳过 OCR 重跑 Formatter）
    python run.py /path/to/images output.epub --save-json intermediate.json

    # 从已有 JSON 跳过 OCR 直接生成 EPUB
    python run.py --from-json intermediate.json output.epub

    # 竖排模式 + 電撃文庫模板
    python run.py /path/to/images output.epub --template denki --vertical

    # 只跑部分 Formatter 步骤
    python run.py /path/to/images output.epub --steps merge_sentences detect_chapters normalize_punct
"""

import argparse
import json
import sys
from pathlib import Path

# 将当前目录加入 path
sys.path.insert(0, str(Path(__file__).parent))

from models.document import UnifiedDocument
from core.temp_manager import TempCropManager


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Novel Formatter Studio — 图片文件夹 → EPUB3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 输入
    parser.add_argument("input", help="图片文件夹路径，或导入模式下的 JSON/EPUB 文件")
    parser.add_argument("output_epub", nargs="?", help="输出 EPUB 路径（批量模式不需要）")

    # 模式
    source_mode = parser.add_mutually_exclusive_group()
    source_mode.add_argument("--from-json", dest="from_json", action="store_true",
                             help="input 参数改为已有的 UnifiedDocument JSON，跳过 OCR")
    source_mode.add_argument("--from-epub", dest="from_epub", action="store_true",
                             help="input 参数改为已有的 .epub 文件，逆向导入后跳过 OCR "
                                  "（常配合 --steps detect_chapters strip_boilerplate 清理样板文字）")

    # OCR 选项
    parser.add_argument("--shortcut", default="ExtractText",
                        help="快捷指令名称（默认: ExtractText）")
    parser.add_argument("--overrides", "-o",
                        help='Page Manager 手动标注 JSON：{"1":"cover","5":"blank",...}')
    parser.add_argument("--crop-top", type=float, default=0.0, metavar="0.0-0.3",
                        help="识别前裁掉页面顶部的比例，排除页眉区域（默认 0 = 不裁）")
    parser.add_argument("--crop-bottom", type=float, default=0.0, metavar="0.0-0.3",
                        help="识别前裁掉页面底部的比例，排除页脚区域（默认 0 = 不裁）")
    parser.add_argument("--text-layer", action="store_true",
                        help="input 是带文字层的 PDF 时，直接读取文字层（更准确、"
                             "自动按字号跳过振假名/页码），不走图像 OCR")

    # 元数据
    parser.add_argument("--title",     help="书名（写入 EPUB metadata）")
    parser.add_argument("--author",    help="作者（写入 EPUB metadata）")
    parser.add_argument("--publisher", help="出版社")
    parser.add_argument("--isbn",      help="ISBN")
    parser.add_argument("--volume",    help="卷号，如 第1巻")

    # Formatter 选项
    from engine.formatter import PIPELINE_STEPS
    step_names = [name for name, _ in PIPELINE_STEPS]
    step_help_lines = "\n".join(
        "  " + "  ".join(step_names[i:i + 4]) for i in range(0, len(step_names), 4)
    )
    parser.add_argument(
        "--steps", nargs="*",
        metavar="STEP",
        help="指定 Formatter 步骤（默认全部，列表自动与 engine.formatter.PIPELINE_STEPS 同步）：\n"
             + step_help_lines,
    )
    parser.add_argument(
        "--repo", metavar="PATH", default=None,
        help="版本仓库目录（默认自动分配临时目录）。指定后每一步 Formatter 都会 "
             "commit 进这个目录，可用 `python -m engine.version_control log --repo PATH` 查看历史",
    )

    # 输出选项
    parser.add_argument(
        "--template", "-t",
        choices=["denki", "mf", "web"],
        default="denki",
        help="CSS 模板（默认: denki = 電撃文庫），仅用于 EPUB 输出",
    )
    parser.add_argument("--vertical", action="store_true", help="竖排模式（默认横排）")
    parser.add_argument("--output-word", metavar="PATH",
                        help="额外生成一份 Word (.docx)，供人工校对/进一步编辑用")

    # 调试
    parser.add_argument("--save-json", metavar="PATH",
                        help="保存 OCR 输出的中间 JSON（方便重跑 Formatter）")
    parser.add_argument("--save-formatted-json", metavar="PATH",
                        help="保存 Formatter 处理后的 JSON")
    parser.add_argument("--batch", action="store_true",
                        help="批量处理输入目录中的书籍文件夹")
    parser.add_argument("--output-dir", default="./output",
                        help="批量输出 EPUB 目录")
    parser.add_argument("--no-preview", action="store_true",
                        help="关闭实时预览/临时裁剪预览输出")

    parser.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        parser.error(f"输入路径不存在: {input_path}")
    if not 0.0 <= args.crop_top <= 0.3:
        parser.error("--crop-top 必须在 0.0 到 0.3 之间")
    if not 0.0 <= args.crop_bottom <= 0.3:
        parser.error("--crop-bottom 必须在 0.0 到 0.3 之间")
    if args.crop_top + args.crop_bottom >= 0.8:
        parser.error("顶部和底部裁剪比例之和必须小于 0.8")

    if args.batch:
        if args.from_json or args.from_epub or args.text_layer:
            parser.error("--batch 不能与 --from-json、--from-epub 或 --text-layer 同时使用")
        if not input_path.is_dir():
            parser.error("批量模式的 input 必须是包含书籍子目录的文件夹")
        from core.batch_processor import BatchProcessor
        BatchProcessor(
            input_dir=args.input,
            output_dir=args.output_dir,
            preview_enabled=not args.no_preview,
        ).run()
        return 0

    if not args.output_epub:
        parser.error("非批量模式必须提供 output_epub")
    output_path = Path(args.output_epub).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet

    # ── Step 1：获取 UnifiedDocument ─────────────────────────────────────────
    if args.from_json:
        if not input_path.is_file():
            parser.error("--from-json 的 input 必须是 JSON 文件")
        if verbose:
            print(f"📥  从 JSON 读取: {args.input}")
        with open(args.input, encoding="utf-8") as f:
            doc = UnifiedDocument.from_json(f.read())
    elif args.from_epub:
        if not input_path.is_file():
            parser.error("--from-epub 的 input 必须是 EPUB 文件")
        from adapters.epub_adapter import import_epub
        doc = import_epub(args.input, verbose=verbose)
        if args.title:     doc.metadata.title     = args.title
        if args.author:    doc.metadata.author    = args.author
        if args.publisher: doc.metadata.publisher = args.publisher
        if args.isbn:      doc.metadata.isbn      = args.isbn
        if args.volume:    doc.metadata.volume    = args.volume
    else:
        overrides = {}
        if args.overrides:
            with open(args.overrides, encoding="utf-8") as f:
                overrides = json.load(f)
            if verbose:
                print(f"📋  Page Manager 标注: {len(overrides)} 页已覆盖")

        if args.text_layer:
            if verbose:
                print(f"\n── 第一步：PDF 文字层提取 ─────────────────────────────")
            from adapters.pdf_text_layer import extract_pdf_text_layer
            doc = extract_pdf_text_layer(
                args.input,
                page_overrides=overrides,
                verbose=verbose,
            )
        else:
            if verbose:
                print(f"\n── 第一步：Apple Vision OCR ─────────────────────────────")
            from adapters.apple_vision_adapter import run as ocr_run
            crop_manager = TempCropManager(name="novel_crops")
            crop_path = crop_manager.create(Path(args.input).name)
            try:
                doc = ocr_run(
                    image_folder=args.input,
                    page_overrides=overrides,
                    shortcut_name=args.shortcut,
                    verbose=verbose,
                    crop_top=args.crop_top,
                    crop_bottom=args.crop_bottom,
                    temp_crop_dir=str(crop_path),
                    preview_enabled=not args.no_preview,
                )
            finally:
                crop_manager.cleanup(crop_path)

        # 补充命令行传入的 metadata
        if args.title:     doc.metadata.title     = args.title
        if args.author:    doc.metadata.author    = args.author
        if args.publisher: doc.metadata.publisher = args.publisher
        if args.isbn:      doc.metadata.isbn      = args.isbn
        if args.volume:    doc.metadata.volume    = args.volume

        if args.save_json:
            with open(args.save_json, "w", encoding="utf-8") as f:
                f.write(doc.to_json())
            if verbose:
                print(f"💾  中间 JSON 已保存: {args.save_json}")

    if verbose:
        print(f"\n── 第二步：Novel Formatter Engine ──────────────────────────")

    # ── Step 2：Formatter Pipeline ───────────────────────────────────────────
    from engine.formatter import run_pipeline
    formatted = run_pipeline(doc, steps=args.steps, verbose=verbose, repo_path=args.repo)
    if verbose and formatted.repo is not None:
        print(f"📚  版本仓库: {formatted.repo.path}（可用 version_control log 查看历史）")

    if args.save_formatted_json:
        with open(args.save_formatted_json, "w", encoding="utf-8") as f:
            f.write(formatted.to_json())
        if verbose:
            print(f"💾  Formatter JSON 已保存: {args.save_formatted_json}")

    if verbose:
        print(f"\n── 第三步：EPUB Builder ──────────────────────────────────────")

    # ── Step 3：EPUB Builder ─────────────────────────────────────────────────
    from builder.epub_builder import build_epub
    build_epub(
        formatted,
        output_path=str(output_path),
        css_template=args.template,
        vertical=args.vertical,
        verbose=verbose,
    )

    # ── 可选：额外生成 Word ───────────────────────────────────────────────────
    if args.output_word:
        from builder.word_builder import build_word
        build_word(
            formatted,
            output_path=args.output_word,
            vertical=args.vertical,
            verbose=verbose,
        )

    if verbose:
        print(f"\n🎉  全部完成！")
        print(f"    输入: {args.input}")
        print(f"    输出: {output_path}")
        if args.output_word:
            print(f"    Word: {args.output_word}")
        toc_items = "\n        ".join(t.title for t in formatted.toc)
        if toc_items:
            print(f"    目录:\n        {toc_items}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
