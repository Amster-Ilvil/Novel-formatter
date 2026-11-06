<div align="center">

**English** · [简体中文](README.zh-CN.md) · [日本語](README_ja.md)

</div>

<p align="center">
  <img src="assets/novel_formatter_banner.webp" alt="Novel Formatter" width="100%">
</p>

<h1 align="center">Novel Formatter</h1>

<p align="center">
  <img alt="macOS" src="[https://img.shields.io/badge/macOS-Apple%20Silicon%20%2F%20Intel-black?logo=apple](https://img.shields.io/badge/macOS-Apple%20Silicon%20%2F%20Intel-black?logo=apple)">
  <img alt="Windows" src="[https://img.shields.io/badge/Windows-x64-0078D4?logo=windows11&logoColor=white](https://img.shields.io/badge/Windows-x64-0078D4?logo=windows11&logoColor=white)">
  <img alt="PySide6" src="[https://img.shields.io/badge/UI-PySide6-41CD52?logo=qt&logoColor=white](https://img.shields.io/badge/UI-PySide6-41CD52?logo=qt&logoColor=white)">
  <img alt="Japanese OCR" src="[https://img.shields.io/badge/OCR-Japanese-2f6feb](https://img.shields.io/badge/OCR-Japanese-2f6feb)">
  <img alt="EPUB" src="[https://img.shields.io/badge/Output-EPUB-8A2BE2](https://img.shields.io/badge/Output-EPUB-8A2BE2)">
</p>

> Japanese vertical OCR · Multi-engine OCR comparison and fusion · Image/text proofreading · Formatter · EPUB · AI review packages · Apple Vision

This repository is a fork of [Amster-Ilvil/Novel-formatter](https://github.com/Amster-Ilvil/Novel-formatter). The fork adds signature-based input detection, an optional left-to-right EPUB conversion, a unified model download manager, recognizer/segmentation performance work and an English-by-default interface. This English README is the reference documentation for those features; the Chinese and Japanese READMEs link back here.

## Contents

- [Overview](#overview)
- [Requirements and installation](#requirements-and-installation)
- [Running](#running)
- [Supported inputs](#supported-inputs)
- [Models: automatic download](#models-automatic-download)
- [Left-to-right (horizontal) conversion](#left-to-right-horizontal-conversion)
- [Interface language](#interface-language)
- [Performance](#performance)
- [Development and tests](#development-and-tests)
- [References and acknowledgments](#references-and-acknowledgments)
- [Use and license](#use-and-license)

## Overview

Novel Formatter turns scanned Japanese light novels (images, image folders, PDFs, comic-style archives) into EPUB 3. The pipeline is:

1. **Import** — pages come from images, folders, PDFs or ZIP/CBZ/RAR/CBR archives; file types are detected by content (see [Supported inputs](#supported-inputs)). Page Management labels cover, title page, table of contents, illustrations, body, afterword and so on.
2. **Recognition** — one or more OCR engines run per page or per physical column; results can be compared, fused and adjudicated against the source image.
3. **Formatting** — the Formatter rebuilds reading order, paragraphs, dialogue, punctuation, ruby, chapters and cross-page continuation.
4. **EPUB export** — EPUB 3 with CSS templates, vertical or horizontal layout, optional left-to-right conversion, plus optional DOCX and AI review packages.

### Main features

- **Image/PDF import:** individual images, image folders, PDF pages (including the PDF text layer) and image archives.
- **Vertical Japanese OCR:** vertical writing, per-column recognition and text spanning columns.
- **Multi-engine OCR comparison and fusion:** review results from Apple Vision, Windows Snipping OCR, NDLOCR-Lite, Hayai OCR, 48px OCR, PaddleOCR and other supported engines.
- **Image/text proofreading:** compare the source page with its OCR output to find wrong characters, omissions, low-confidence text and layout problems.
- **Ruby, headers and page numbers:** FindText CenterNet helps detect ruby/furigana areas and remove running headers, page numbers and fragments across columns or pages.
- **Formatter:** clean up paragraphs, punctuation, headings and cross-page continuation after OCR.
- **EPUB creation:** generate EPUB from corrected text, book structure and embedded resources.
- **AI review packages:** export evidence, text, structure and resources for further review with GPT, Gemini, GLM, DeepSeek and compatible services.
- **Local data first:** user documents, model weights, caches, logs and databases are not committed to the repository.

### What's new in v2.0 (upstream)

Compared with v1.3, v2.0 expands the OCR, proofreading, PDF and EPUB workflows:

- **New project workspace:** pages, OCR results, decisions, formatted text and EPUB production state live in one workspace; restore work and browse recent projects.
- **Multi-engine OCR and adjudication:** per-column and full-page recognition roles, disagreement review, image/text comparison and manual decisions; improved integration with Hayai OCR, NDLOCR-Lite and 48px OCR.
- **Native system OCR:** Windows 11 Snipping OCR and a more robust native Apple Vision path on Intel and Apple Silicon Macs.
- **PDF text layer:** uses original PDFs and physical-page mappings from Page Management for better vertical reading order, cross-page continuation and paragraphs.
- **AI review improvements:** GPT, Gemini, GLM, DeepSeek and compatible providers, with safe round-tripping of packages, OCR evidence and manual edits.
- **EPUB/TEI and UI:** cover images, illustrations, page boundaries, metadata, EPUB structure, previews, progress reporting and stability.

### Recommended workflow

1. Import a PDF, image, image folder or archive.
2. Label the cover, title page, table of contents, illustrations, body pages, afterword and other page types.
3. Choose the OCR engines and optionally enable column-by-column recognition.
4. Run OCR.
5. Inspect disagreements, missing text, ruby, page numbers and cross-page continuation in OCR Compare and Image/Text Review.
6. Apply the fused result and format the text with the Formatter.
7. Export EPUB directly or export an AI review package for more proofreading.

### AI review packages

AI review packages combine OCR evidence, text, structure and resources, so a model can check the original context rather than an isolated transcript. They help with correcting OCR mistakes, missing text and punctuation; checking names, places and terms for consistency; reviewing context across pages and chapters; checking EPUB structure and typography; and producing a more publication-ready EPUB.

## Requirements and installation

### Python

The repository has no `pyproject.toml`/`setup.py` and the code does not enforce a minimum version. The upstream README recommends **Python 3.10 or later**; the release workflows in `.github/workflows/` build with **Python 3.12**. This fork's branch is developed and tested with Python 3.14.

### Python packages

| File | Purpose |
|---|---|
| `requirements.txt` | Core: `PySide6>=6.7,<7`, `Pillow>=10,<13`, `PyMuPDF>=1.24,<2`, `fonttools`, `python-docx`, `pdf2image`, `pillow-heif>=0.18,<2`; AI providers `httpx`, `openai`, `anthropic`, `google-generativeai`; matching/tokens `diff-match-patch`, `rapidfuzz`, `tiktoken`; `epubcheck>=5.3,<6` for EPUB validation |
| `requirements-handwriting-openvino.txt` | Optional independent handwriting recognizer: `numpy>=1.26,<3`, `openvino>=2024.0,<2027` |

HEIC/HEIF support (`pillow-heif`) is part of the core requirements, not an optional extra.

### macOS

```bash
git clone https://github.com/Amster-Ilvil/Novel-formatter.git
cd Novel-formatter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python gui_pyside6.py
```

Alternatively run `./run_novel_formatter.command`.

**Apple Vision helper.** The native Apple Vision OCR path uses a small Swift helper built by `build_apple_vision_helper.command` from `tools/apple_vision_helper/AppleVisionOCRHelper.swift`. It needs `swiftc` from Xcode or the Command Line Tools (`xcode-select --install`), targets macOS 13.0 by default (`MACOSX_DEPLOYMENT_TARGET`) and writes the binary to `~/Library/Caches/NovelFormatter/apple_vision_helper/<arch>/apple_vision_helper` (override with `NOVEL_FORMATTER_APPLE_VISION_HELPER_DIR`). The binary is ad-hoc signed with `codesign` when available.

```bash
./build_apple_vision_helper.command
```

Apple Vision, the Swift OCR helper and Apple Pencil handwriting recognition are macOS-only.

### Windows

```powershell
git clone https://github.com/Amster-Ilvil/Novel-formatter.git
cd Novel-formatter
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python gui_pyside6.py
```

Windows Snipping OCR requires compatible Windows 11 system OCR components. Prebuilt upstream installers are available from [Releases](https://github.com/Amster-Ilvil/Novel-formatter/releases).

### Optional: RAR/CBR archives

RAR support is optional. It needs the `rarfile` Python package **and** an `unrar` or `bsdtar` binary on `PATH`:

```bash
python -m pip install rarfile
brew install libarchive      # provides bsdtar on macOS; or install unrar
```

Without them, a RAR/CBR input fails with `RAR support requires the optional `rarfile` package and an unrar/bsdtar binary`; all other formats keep working.

### Optional OCR engines

Engines other than the system OCR run in their own virtual environments, which the app creates on first use after confirmation (**Install and Continue**). Detection (`adapters/ocr_runtime_catalog.py`) never creates a venv and never downloads a model.

| Component id | Engine | Environment |
|---|---|---|
| `paddle_ocr` | PaddleOCR / PP-OCR (`japan` for vertical Japanese, `ch` for horizontal Simplified Chinese) | `.venv-paddle` |
| `paddle_structure` | PP-StructureV3 (installs `paddlex[ocr]`, larger layout models) | `.venv-paddle` |
| `paddle_vl` | PaddleOCR-VL-1.6 (Apple Silicon uses the official MLX-VLM backend in a separate `.venv-mlx-vlm`, falls back to native Paddle) | `.venv-paddle` |
| `ndlocr_lite` | NDLOCR-Lite (official source, ONNX models) | `.venv-ndlocr-lite` |
| `findtext_centernet_ruby` | findtextCenterNet ruby expert | `.venv-findtext-centernet` |
| `hayai_ocr` | Hayai OCR · PyTorch (MPS/CUDA/CPU, optional INT4/INT8) | shared persistent Hayai venv |
| `hayai_ocr_litert` | Hayai OCR v2 · LiteRT (extra LiteRT runtime and TFLite weights) | shared persistent Hayai venv |
| `manga_48px` | Manga 48px AR OCR (weights via the model manager) | `.venv-manga-48px` |

The OpenVINO handwriting recognizer uses `requirements-handwriting-openvino.txt` and the model manager. Apple Vision (macOS) and Windows Snipping OCR (Windows 11) use the operating system.

## Running

### GUI

```bash
.venv/bin/python gui_pyside6.py
```

### CLI: full pipeline (`run.py`)

`run.py` runs image folder → OCR → Formatter → EPUB 3 in one shot. Its OCR path calls Apple Vision through macOS Shortcuts (`--shortcut`, default `ExtractText`), so image OCR from the CLI is macOS-only; `--from-json` and `--from-epub` skip OCR.

```bash
# image folder → EPUB
python run.py /path/to/images output.epub --title "Title" --author "Author"

# Page Manager page types, keep the intermediate JSON
python run.py /path/to/images output.epub --overrides overrides.json --save-json intermediate.json

# rebuild from an existing JSON without OCR, left-to-right layout
python run.py --from-json intermediate.json output.epub --ltr

# offline: never download models, use a specific models folder
python run.py /path/to/images output.epub --no-download --models-dir /data/nf-models

# batch: every subfolder of the input is a book
python run.py /path/to/books --batch --output-dir /path/to/out
```

Options (from `python run.py --help`):

| Option | Meaning |
|---|---|
| `input` / `output_epub` | image folder (or JSON/EPUB in import modes) / output EPUB path (not needed with `--batch`) |
| `--from-json` \| `--from-epub` | treat input as UnifiedDocument JSON or an existing `.epub` and skip OCR |
| `--shortcut SHORTCUT` | macOS Shortcuts name (default `ExtractText`) |
| `--overrides, -o OVERRIDES` | Page Manager page types as JSON, e.g. `{"1":"cover","5":"blank"}` |
| `--crop-top`, `--crop-bottom` `0.0-0.3` | fraction of the page to crop before OCR; the sum must be < 0.8 |
| `--text-layer` | read the PDF text layer instead of image OCR |
| `--preserve-ruby` | additionally keep source ruby via findtextCenterNet (off by default) |
| `--ruby-scan-mode {smart_roi,full_page,auto}` | ruby scan scope (default `smart_roi`) |
| `--title`, `--author`, `--publisher`, `--isbn`, `--volume` | EPUB metadata |
| `--steps [STEP ...]` | Formatter steps to run (default: all of `engine.formatter.PIPELINE_STEPS`) |
| `--repo PATH` | version repository; each Formatter step commits into it (`python -m engine.version_control log --repo PATH`) |
| `--template, -t {denki,mf,web}` | CSS template (default `denki`) |
| `--vertical` | vertical mode (default: horizontal) |
| `--ltr` | convert to left-to-right horizontal layout before export |
| `--output-word PATH` | also write a `.docx` |
| `--save-json PATH`, `--save-formatted-json PATH` | save the OCR JSON / the JSON after Formatter |
| `--batch`, `--output-dir OUTPUT_DIR` | batch-process book folders; output folder for EPUBs |
| `--no-preview`, `--quiet, -q` | disable live/crop preview output; less output |
| `--models-dir PATH` | models folder (takes precedence over `NOVEL_FORMATTER_MODELS_DIR`) |
| `--no-download` | never download models; fail with a hint where to place them |

`--batch` cannot be combined with `--from-json`, `--from-epub` or `--text-layer`, and rejects the single-book options `--overrides`, `--preserve-ruby`, `--title`, `--author`, `--publisher`, `--isbn`, `--volume`, `--repo`, `--output-word`, `--save-json`, `--save-formatted-json` and `--ltr` with an error instead of ignoring them.

### CLI: JSON → EPUB only (`builder.epub_builder`)

```bash
python -m builder.epub_builder formatted.json output.epub --template web --ltr
```

| Option | Meaning |
|---|---|
| `input_json` / `output_epub` | JSON produced by the Formatter / output EPUB path |
| `--template, -t {denki,mf,web}` | CSS template (default `denki`) |
| `--vertical` | vertical mode (default: horizontal) |
| `--ltr` | convert to left-to-right (horizontal) layout before export |
| `--quiet, -q` | less output |

## Supported inputs

Input types are detected from the first 8 KiB of each file (`adapters/file_type.py`, stdlib only). The extension is used only when no signature matches; such results are tagged `by-extension`.

| Kind | Formats | Detection notes | Used as pages |
|---|---|---|---|
| Image | PNG, JPEG, GIF, BMP, TIFF, WebP, HEIC/HEIF | magic bytes; BMP validated by DIB header size; HEIF by `ftyp` brand | yes |
| Image | PPM | by `.ppm` extension only (no signature check) | yes |
| PDF | PDF | `%PDF-` anywhere in the first 1024 bytes (tolerates leading junk) | yes (rendered pages or `--text-layer`) |
| ZIP archive | `.zip`, `.cbz` | ZIP signature; EPUB is distinguished from plain ZIP | yes (images inside) |
| RAR archive | `.rar`, `.cbr` | RAR signature; extraction needs optional `rarfile` + `unrar`/`bsdtar` | yes (images inside) |
| EPUB | `.epub` | detected separately from ZIP | no (use `run.py --from-epub`) |
| Text | `.txt`, `.md`, `.markdown` | text heuristic (control-character ratio) | no |

Behaviour:

- **Wrong or missing extension:** content wins. A PNG saved as `.jpg` or a file without a suffix is imported as what it really is, and extracted archive pages get the extension of the sniffed signature. `.cbz`/`.cbr` keep their suffix and are shown as comic archives.
- **Unsupported files:** when none of the selected inputs is usable, import raises `UnsupportedFileTypeError` listing every rejected path with its detected type, followed by `Supported: images (PNG/JPEG/GIF/BMP/TIFF/WebP/HEIC), PDF, ZIP/CBZ and RAR/CBR image archives.` When at least one input is usable, the rest (text, EPUB, unknown binaries, broken or image-less archives, missing paths) is reported in the skipped-files log and the import continues.
- **Folders** are scanned one level deep in natural order (`p2` before `p10`); hidden entries and AppleDouble `._*` files are skipped silently.
- **Archives:** entries are sorted naturally by their full member path and written as `00001_<stem><ext>`. Directories, `__MACOSX/` and hidden `.`/`._` entries are ignored silently. Unsafe entries (`..` or absolute paths — zip-slip — symlinks, encrypted entries) and non-image members are reported as skipped. Limits: 20 000 entries and 4 GiB uncompressed; an archive without any image is an error.

## Models: automatic download

Downloadable OCR weights go through one code path, `utils/model_manager.py`: locate an installed copy → verify → otherwise download → verify → atomically install → use.

### Registry

Data from `MODEL_REGISTRY` in `utils/model_manager.py`:

| Model id | Engine | File | Size (bytes) | SHA-256 | SHA-384 | `hash_source` |
|---|---|---|---:|:-:|:-:|---|
| `manga_48px.checkpoint` | `manga_48px` | `ocr_ar_48px.ckpt` | 204,290,192 | ✓ | — | `existing-code` |
| `manga_48px.dictionary` | `manga_48px` | `alphabet-all-v7.txt` | not pinned (≥ 1,000) | ✓ | — | `existing-code` |
| `openvino_handwriting.xml` | `openvino_handwriting` | `handwritten-japanese-recognition-0001.xml` | 97,226 | ✓ | ✓ | `sha384: existing-code; sha256: pinned-from-official-download 2026-11-05` |
| `openvino_handwriting.bin` | `openvino_handwriting` | `handwritten-japanese-recognition-0001.bin` | 17,035,726 | ✓ | ✓ | `sha384: existing-code; sha256: pinned-from-official-download 2026-11-05` |
| `openvino_handwriting.charlist` | `openvino_handwriting` | `kondate_nakayosi.txt` | 17,711 | ✓ | — | `pinned-from-official-download 2026-11-05` |

Sources: the 48px files come from Hugging Face `zyddnys/manga-image-translator` at a pinned revision, with the GitHub release `beta-0.3` as a mirror; the OpenVINO model from `storage.openvinotoolkit.org` (Open Model Zoo 2023.0, FP16-INT8); the character list from `openvinotoolkit/open_model_zoo` at a pinned commit.

Not in the registry:

- **NDLOCR-Lite** ONNX models ship with the pinned upstream source tag. Upstream publishes no hashes, so only presence, size and "not a Git LFS pointer" are checked.
- **PaddleOCR** models are downloaded by PaddleX itself. Only when a models folder is set explicitly (Settings, `--models-dir` or `NOVEL_FORMATTER_MODELS_DIR`) is `PADDLE_PDX_CACHE_HOME` pointed at `<models folder>/paddlex`.

### Where models are stored

`resolve_models_dir()` picks the first match:

1. **Settings → OCR → Local models folder**, or `run.py --models-dir PATH`;
2. environment variable `NOVEL_FORMATTER_MODELS_DIR`;
3. the legacy project cache `.model-cache/` — only if it already exists;
4. the per-user cache:

| Platform | Default models folder |
|---|---|
| macOS | `~/Library/Caches/Novel-formatter/models` |
| Windows | `%LOCALAPPDATA%\Novel-formatter\models` (fallback `~\AppData\Local\Novel-formatter\models`) |
| Linux/other | `$XDG_CACHE_HOME/novel-formatter/models`, or `~/.cache/novel-formatter/models` |

Each engine has a subfolder: `manga-48px-ar/` for 48px, `handwriting_openvino/` for OpenVINO. Existing installations in the legacy locations (`.model-cache/manga-48px-ar`, `models/handwriting_openvino` in the project) are found first and are not downloaded again.

### First run, verification and failures

- On first use of an engine the missing files are downloaded with a progress dialog and a **Cancel** button. Data goes to `<file>.part` in 1 MiB blocks with timeouts, retries with backoff and mirror fallback; an interrupted download resumes via HTTP `Range`.
- After download the size and SHA-256 (plus SHA-384 where pinned) are checked, then the file is moved into place with an atomic `os.replace`. A corrupt or partial file never stays under its final name. A `.verified` marker keyed on size, mtime and digest avoids re-hashing large checkpoints on every start.
- Concurrent callers are serialised per model with a thread lock plus an inter-process lock file.
- Errors are raised as `ModelError` subclasses with the model id, URL and target folder:
  - `ModelNetworkError` — no network, DNS failure, refused connection or timeout;
  - `ModelDownloadError` — the server answered with an HTTP error status;
  - `ModelIntegrityError` — size, hash, archive layout or archive safety check failed (the file is discarded);
  - `ModelCancelledError` — cancelled by the user;
  - `ModelNotAvailableError` — the model is missing and downloading is not allowed.

### Offline use and manual installation

- **Settings → OCR → Download models automatically when needed** is on by default. Turning it off, passing `run.py --no-download` or setting `NOVEL_FORMATTER_MODEL_AUTO_DOWNLOAD=0` (also `false`, `no`, `off`) disables automatic downloads; a missing model then fails with `ModelNotAvailableError` naming the download URL and target folder. Explicit "Download OCR Models" buttons still work when automatic downloads are off.
- **Manual installation:** download the files from the URLs listed in `MODEL_REGISTRY` and place them under the models folder, keeping the file names from the table, e.g. `<models folder>/manga-48px-ar/ocr_ar_48px.ckpt` and `<models folder>/handwriting_openvino/handwritten-japanese-recognition-0001.xml`. They are verified against the pinned hashes on next use.

## Left-to-right (horizontal) conversion

An optional export step (`builder/text_direction.py`) turns a vertical Japanese EPUB into a horizontal left-to-right publication.

**Where to enable it:**

- GUI: **EPUB** tab → **Convert to left-to-right (horizontal) layout**;
- CLI: `run.py ... --ltr` or `python -m builder.epub_builder ... --ltr`;
- API: `build_epub(..., text_direction="ltr")`; the default is `"keep"` (`TextDirectionMode.KEEP`).

**What it does:**

- sets `page-progression-direction="ltr"` on the spine, `dir="ltr"` on documents and `writing-mode: horizontal-tb`;
- removes vertical-only CSS rules;
- replaces vertical presentation forms of punctuation (U+FE10–FE19, U+FE30–FE4F) with their ordinary forms;
- unwraps tate-chū-yoko (horizontal-in-vertical) spans while keeping ruby;
- swaps the halves of detected two-page illustration spreads so the left half comes first.

**What it does not do:**

- it does not reorder pages or chapters;
- it does not re-run OCR or the Formatter.

**Guarantees:**

- With `KEEP` (the default) the builder output is byte-for-byte identical to the output before this option existed.
- Content that is already horizontal only gets `dir`/spine direction; the conversion is idempotent.
- `--ltr` is rejected in `run.py --batch` mode.

## Interface language

The interface is **English by default** on a fresh install. A previously saved Simplified Chinese (`zh_CN`) or Japanese (`ja_JP`) choice is kept. Switch between **English / 简体中文 / 日本語** in **Settings → Interface & Appearance → Interface Language**; the change applies without restarting and affects only UI chrome, never OCR text, documents or model data. CLI messages are in English.

Translations are offline and explicit (unknown strings stay unchanged rather than being machine-translated):

- `ui/localization.py`: language codes (`SUPPORTED_LANGUAGES`, `LANGUAGE_LABELS`, `DEFAULT_LANGUAGE`) and the main exact-match dictionaries (`_EXACT_EN`, `_EXACT_JA`);
- `ui/dialog_localization.py`: dialog strings;
- `ui/localization_audit_catalog.py`: catalog of strings found by the audit, including backend messages.

To add a language: add its code to `SUPPORTED_LANGUAGES` and a label to `LANGUAGE_LABELS` in `ui/localization.py`, then add dictionaries alongside the existing English/Japanese ones in the three files above, following the same pattern. Use the audit tool to find strings that are still missing:

```bash
python tools/i18n_audit.py --stats             # per-file counts of CJK UI strings without English
python tools/i18n_audit.py ui/settings         # list them for specific files/dirs
python tools/i18n_audit.py --backend --stats   # exception/log/progress text in adapters, engine, builder, core, utils
python tools/i18n_audit.py --backend-visible --stats  # user-visible (A) vs internal (B) literals in adapters/ and engine/
```

## Performance

What was optimized:

- **Recognizer session cache** (`adapters/recognizer_session_cache.py`): recognizer sessions are reused between jobs; **Stop** does not discard them.
- **Single-pass segmentation:** connected-component column segmentation in one pass instead of several.
- Equivalence references pin the old outputs (see [Development and tests](#development-and-tests)).

Measured with `tools/bench_pipeline.py` on 42 synthetic pages at 200 DPI, Apple M3 Pro:

| Measurement | Before | After | Speedup |
|---|---:|---:|---:|
| End-to-end, cold | 17.654 s | 16.692 s | ×1.06 |
| End-to-end, warm | 17.254 s | 16.358 s | ×1.05 |
| Segmentation stage | — | — | ×1.31 |

> **Caveat:** synthetic pages and a **fake OCR engine**. The numbers show pipeline overhead only; the speedup with real OCR engines has not been measured.

Running the benchmark:

```bash
# generate the deterministic fixture PDF only
python tools/bench_pipeline.py --generate-only

# benchmark a checkout and write a report; repeat for another checkout
python tools/bench_pipeline.py --repo /path/to/checkout --label after --runs 3 --out-dir /tmp/nf-bench

# compare two reports
python tools/bench_pipeline.py --compare before.json after.json
```

Other options: `--pdf PDF` (input PDF, generated if missing), `--dpi DPI`, `--skip-stages` (end-to-end only), `--skip-e2e` (per-stage loop only), `--no-isolate-home` (keep the real `HOME`; persistent user caches may skew results).

## Development and tests

```bash
.venv/bin/python -m pytest -q
```

- UI tests (`tests/test_ui_smoke.py`, `tests/test_model_download_ui.py`) set `QT_QPA_PLATFORM=offscreen` themselves (via `os.environ.setdefault`) and are skipped when PySide6 is not installed, so no display is needed.
- `tests/test_performance_equivalence.py` compares current behaviour with references captured **before** the optimizations: `tests/fixtures/segmentation_reference.json` (from commit `b962ff0`, `--write`) and `tests/fixtures/column_image_reference.json` (from commit `858427f`, `--write-columns`; exact pixels of every per-column OCR input).
- **Do not regenerate the `tests/fixtures/*reference*.json` files to make a failing test pass.** They exist to prove that performance changes did not change the output. If the test fails, the change altered segmentation or column images; fix the code, or—if the change is intended—regenerate the reference from the documented baseline procedure in the test's docstring and explain it in the commit.

## References and acknowledgments

The UI, OCR integrations, document processing and handwriting recognition use or draw on the following projects and documentation:

- [Qt for Python / PySide6](https://doc.qt.io/qtforpython/): desktop GUI framework.
- [Apple Vision](https://developer.apple.com/documentation/vision): macOS-native OCR and vision APIs.
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): general OCR and cross-platform reference.
- [NDLOCR-Lite](https://github.com/ndl-lab/ndlocr-lite): OCR and layout analysis for Japanese books.
- [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo): optional handwriting models and inference examples.
- [PyMuPDF](https://pymupdf.readthedocs.io/): PDF pages, images and text layers.
- [python-docx](https://python-docx.readthedocs.io/): read and generate DOCX.
- [Pillow](https://python-pillow.org/): image loading, cropping and preprocessing.
- [jlect-jhr](https://github.com/ZacharyRead/jlect-jhr): Japanese handwriting resources; see `third_party/jlect_jhr/LICENSE.txt`.

Third-party projects, models and resources have their own licenses. No OCR model weights are redistributed in this repository. See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for full credits.

## Star History

<p align="center">
  <a href="[https://www.star-history.com/?type=date&repos=Amster-Ilvil%2FNovel-formatter](https://www.star-history.com/?type=date&repos=Amster-Ilvil%2FNovel-formatter)">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="[https://api.star-history.com/svg?repos=Amster-Ilvil/Novel-formatter&type=Date&theme=dark](https://api.star-history.com/svg?repos=Amster-Ilvil/Novel-formatter&type=Date&theme=dark)">
      <source media="(prefers-color-scheme: light)" srcset="[https://api.star-history.com/svg?repos=Amster-Ilvil/Novel-formatter&type=Date](https://api.star-history.com/svg?repos=Amster-Ilvil/Novel-formatter&type=Date)">
      <img alt="Novel Formatter Star History Chart" src="[https://api.star-history.com/svg?repos=Amster-Ilvil/Novel-formatter&type=Date](https://api.star-history.com/svg?repos=Amster-Ilvil/Novel-formatter&type=Date)" width="800">
    </picture>
  </a>
</p>

## Use and license

This is a fork of [Amster-Ilvil/Novel-formatter](https://github.com/Amster-Ilvil/Novel-formatter). The project's original code and documentation are available under the [MIT License](LICENSE) for personal, educational, research and commercial use, subject to its terms.

Third-party OCR engines, models, dependencies and platform capabilities remain governed by their respective licenses and terms. See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for full credits.
