#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Novel Formatter Studio — one-shot command-line runner

Full pipeline:
    image folder
        → Apple Vision OCR (via macOS Shortcuts)
        → unified document model JSON (intermediate result can be saved)
        → Novel Formatter Engine (post-processing steps)
        → EPUB3 output

Usage:
    # Minimal: image folder → EPUB
    python run.py /path/to/images output.epub

    # Set title and author
    python run.py /path/to/images output.epub --title "魔法科高校の劣等生" --author "佐島勤"

    # Use Page Manager manual page types (JSON)
    python run.py /path/to/images output.epub --overrides overrides.json

    # Save the intermediate JSON (for debugging or re-running Formatter without OCR)
    python run.py /path/to/images output.epub --save-json intermediate.json

    # Skip OCR and build the EPUB from an existing JSON
    python run.py --from-json intermediate.json output.epub

    # Vertical mode + Dengeki Bunko template
    python run.py /path/to/images output.epub --template denki --vertical

    # Run only some Formatter steps
    python run.py /path/to/images output.epub --steps merge_sentences detect_chapters normalize_punct
"""

import argparse
import json
import sys
from pathlib import Path

# Make the project root importable.
sys.path.insert(0, str(Path(__file__).parent))

from models.document import UnifiedDocument
from core.temp_manager import TempCropManager


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Novel Formatter Studio — image folder → EPUB3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    parser.add_argument("input", help="image folder, or a JSON/EPUB file in import modes")
    parser.add_argument("output_epub", nargs="?", help="output EPUB path (not needed in batch mode)")

    # Source mode
    source_mode = parser.add_mutually_exclusive_group()
    source_mode.add_argument("--from-json", dest="from_json", action="store_true",
                             help="treat input as an existing UnifiedDocument JSON and skip OCR")
    source_mode.add_argument("--from-epub", dest="from_epub", action="store_true",
                             help="treat input as an existing .epub, reverse-import it and skip OCR "
                                  "(often combined with --steps detect_chapters strip_boilerplate to clean boilerplate)")

    # OCR options
    parser.add_argument("--shortcut", default="ExtractText",
                        help="macOS Shortcuts name (default: ExtractText)")
    parser.add_argument("--overrides", "-o",
                        help='Page Manager manual page types as JSON: {"1":"cover","5":"blank",...}')
    parser.add_argument("--crop-top", type=float, default=0.0, metavar="0.0-0.3",
                        help="fraction of the page top to crop before OCR to drop running headers (default 0 = no crop)")
    parser.add_argument("--crop-bottom", type=float, default=0.0, metavar="0.0-0.3",
                        help="fraction of the page bottom to crop before OCR to drop footers (default 0 = no crop)")
    parser.add_argument("--text-layer", action="store_true",
                        help="if input is a PDF with a text layer, read the text layer directly (more accurate; "
                             "furigana/page numbers are skipped by font size automatically) instead of image OCR")
    parser.add_argument("--preserve-ruby", action="store_true",
                        help="additionally use findtextCenterNet to keep the Japanese source Ruby; off by default. "
                             "Only extracts the furigana structure and never overwrites the main OCR body text")
    parser.add_argument(
        "--ruby-scan-mode", choices=["smart_roi", "full_page", "auto"], default="smart_roi",
        help="Ruby scan scope: smart_roi = only the suspected Ruby context boxes recorded by regular OCR (default); "
             "full_page = precise full-page scan (slow); auto = ROI when candidates exist, otherwise full page",
    )

    # Metadata
    parser.add_argument("--title",     help="book title (written to EPUB metadata)")
    parser.add_argument("--author",    help="author (written to EPUB metadata)")
    parser.add_argument("--publisher", help="publisher")
    parser.add_argument("--isbn",      help="ISBN")
    parser.add_argument("--volume",    help="volume label, e.g. Vol. 1")

    # Formatter options
    from engine.formatter import PIPELINE_STEPS
    step_names = [name for name, _ in PIPELINE_STEPS]
    step_help_lines = "\n".join(
        "  " + "  ".join(step_names[i:i + 4]) for i in range(0, len(step_names), 4)
    )
    parser.add_argument(
        "--steps", nargs="*",
        metavar="STEP",
        help="Formatter steps to run (default: all; the list is synced with engine.formatter.PIPELINE_STEPS):\n"
             + step_help_lines,
    )
    parser.add_argument(
        "--repo", metavar="PATH", default=None,
        help="version repository folder (default: a temporary folder). When set, every Formatter step "
             "commits into it; view the history with `python -m engine.version_control log --repo PATH`",
    )

    # Output options
    parser.add_argument(
        "--template", "-t",
        choices=["denki", "mf", "web"],
        default="denki",
        help="CSS template (default: denki = Dengeki Bunko); used only for EPUB output",
    )
    parser.add_argument("--vertical", action="store_true", help="vertical mode (default: horizontal)")
    parser.add_argument("--ltr", action="store_true",
                        help="convert to left-to-right horizontal layout before exporting the EPUB")
    parser.add_argument("--output-word", metavar="PATH",
                        help="also write a Word (.docx) file for manual proofreading/further editing")

    # Debugging
    parser.add_argument("--save-json", metavar="PATH",
                        help="save the intermediate OCR JSON (to re-run Formatter later)")
    parser.add_argument("--save-formatted-json", metavar="PATH",
                        help="save the JSON after Formatter processing")
    parser.add_argument("--batch", action="store_true",
                        help="batch-process book folders inside the input folder")
    parser.add_argument("--output-dir", default="./output",
                        help="output folder for EPUBs in batch mode")
    parser.add_argument("--no-preview", action="store_true",
                        help="disable live preview / temporary crop preview output")

    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--models-dir", metavar="PATH", default=None,
                        help="local model folder (takes precedence over NOVEL_FORMATTER_MODELS_DIR) for finding/downloading OCR models")
    parser.add_argument("--no-download", action="store_true",
                        help="never download models automatically; fail with a hint about where to place missing models")

    args = parser.parse_args(argv)
    if args.models_dir or args.no_download:
        from utils.model_manager import configure as configure_models
        # Mirrored into env vars so OCR worker subprocesses use the same settings.
        configure_models(
            models_dir=args.models_dir,
            auto_download=False if args.no_download else None,
        )

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        parser.error(f"input path does not exist: {input_path}")
    if not 0.0 <= args.crop_top <= 0.3:
        parser.error("--crop-top must be between 0.0 and 0.3")
    if not 0.0 <= args.crop_bottom <= 0.3:
        parser.error("--crop-bottom must be between 0.0 and 0.3")
    if args.crop_top + args.crop_bottom >= 0.8:
        parser.error("the sum of top and bottom crop fractions must be less than 0.8")

    if args.batch:
        if args.from_json or args.from_epub or args.text_layer:
            parser.error("--batch cannot be combined with --from-json, --from-epub or --text-layer")
        unsupported = []
        for flag, value in (
            ("--overrides", args.overrides), ("--preserve-ruby", args.preserve_ruby),
            ("--title", args.title), ("--author", args.author), ("--publisher", args.publisher),
            ("--isbn", args.isbn), ("--volume", args.volume), ("--repo", args.repo),
            ("--output-word", args.output_word), ("--save-json", args.save_json),
            ("--save-formatted-json", args.save_formatted_json),
            ("--ltr", args.ltr),
        ):
            if value:
                unsupported.append(flag)
        if unsupported:
            parser.error(
                "batch mode does not support these single-book options yet (they used to be ignored silently): "
                + ", ".join(unsupported)
            )
        if not input_path.is_dir():
            parser.error("in batch mode, input must be a folder containing book subfolders")
        from core.batch_processor import BatchProcessor
        BatchProcessor(
            input_dir=args.input,
            output_dir=args.output_dir,
            preview_enabled=not args.no_preview,
            vertical=args.vertical,
            css_template=args.template,
            shortcut_name=args.shortcut,
            crop_top=args.crop_top,
            crop_bottom=args.crop_bottom,
            steps=args.steps,
            verbose=not args.quiet,
        ).run()
        return 0

    if not args.output_epub:
        parser.error("output_epub is required outside batch mode")
    output_path = Path(args.output_epub).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet

    # ── Step 1: obtain the UnifiedDocument ──────────────────────────────────
    if args.from_json:
        if not input_path.is_file():
            parser.error("with --from-json, input must be a JSON file")
        if verbose:
            print(f"📥  Reading JSON: {args.input}")
        with open(args.input, encoding="utf-8") as f:
            doc = UnifiedDocument.from_json(f.read())
    elif args.from_epub:
        if not input_path.is_file():
            parser.error("with --from-epub, input must be an EPUB file")
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
                print(f"📋  Page Manager overrides: {len(overrides)} page(s)")

        if args.text_layer:
            if verbose:
                print(f"\n── Step 1: PDF text-layer extraction ─────────────────────")
            from adapters.pdf_text_layer import extract_pdf_text_layer
            doc = extract_pdf_text_layer(
                args.input,
                page_overrides=overrides,
                verbose=verbose,
            )
        else:
            if verbose:
                print(f"\n── Step 1: Apple Vision OCR ──────────────────────────────")
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

        if args.preserve_ruby and not args.text_layer:
            if verbose:
                print("\n── Ruby side pass: findtextCenterNet ─────────────────────────")
            from adapters.findtext_centernet_ruby import preserve_ruby_in_documents
            report = preserve_ruby_in_documents(
                [doc],
                log_callback=(print if verbose else None),
                progress_callback=(
                    (lambda current, total, name: print(f"  [Ruby {current}/{total}] {name}"))
                    if verbose else None
                ),
                scan_mode=args.ruby_scan_mode,
            )
            if verbose:
                if report.error:
                    print(f"⚠️ Ruby preservation failed; body text left unchanged: {report.error}")
                else:
                    print(f"✅ Ruby: detected {report.ruby_pairs}, safely written back {report.matched_pairs}")

        # Apply metadata passed on the command line.
        if args.title:     doc.metadata.title     = args.title
        if args.author:    doc.metadata.author    = args.author
        if args.publisher: doc.metadata.publisher = args.publisher
        if args.isbn:      doc.metadata.isbn      = args.isbn
        if args.volume:    doc.metadata.volume    = args.volume

        if args.save_json:
            with open(args.save_json, "w", encoding="utf-8") as f:
                f.write(doc.to_json())
            if verbose:
                print(f"💾  Intermediate JSON saved: {args.save_json}")

    if verbose:
        print(f"\n── Step 2: Novel Formatter Engine ────────────────────────")

    # ── Step 2: Formatter pipeline ──────────────────────────────────────────
    from engine.formatter import run_pipeline
    formatted = run_pipeline(doc, steps=args.steps, verbose=verbose, repo_path=args.repo)
    if verbose and formatted.repo is not None:
        print(f"📚  Version repository: {formatted.repo.path} (see history with version_control log)")

    if args.save_formatted_json:
        with open(args.save_formatted_json, "w", encoding="utf-8") as f:
            f.write(formatted.to_json())
        if verbose:
            print(f"💾  Formatter JSON saved: {args.save_formatted_json}")

    if verbose:
        print(f"\n── Step 3: EPUB Builder ──────────────────────────────────")

    # ── Step 3: EPUB Builder ────────────────────────────────────────────────
    from builder.epub_builder import build_epub
    build_epub(
        formatted,
        output_path=str(output_path),
        css_template=args.template,
        vertical=args.vertical,
        verbose=verbose,
        text_direction="ltr" if args.ltr else "keep",
    )

    # ── Optional: also write a Word document ────────────────────────────────
    if args.output_word:
        from builder.word_builder import build_word
        build_word(
            formatted,
            output_path=args.output_word,
            vertical=args.vertical,
            verbose=verbose,
        )

    if verbose:
        print(f"\n🎉  All done!")
        print(f"    Input:  {args.input}")
        print(f"    Output: {output_path}")
        if args.output_word:
            print(f"    Word:   {args.output_word}")
        toc_items = "\n        ".join(t.title for t in formatted.toc)
        if toc_items:
            print(f"    Contents:\n        {toc_items}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
