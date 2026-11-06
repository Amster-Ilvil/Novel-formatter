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

This is a locally modified build based on [Amster-Ilvil/Novel-formatter](https://github.com/Amster-Ilvil/Novel-formatter). The modifications add signature-based input detection, optional left-to-right EPUB conversion, a model download manager, recognizer/segmentation performance work and an English-by-default interface. This English README is the reference documentation for those features; the Chinese and Japanese READMEs link back here. No published remote for this modified build is established by this documentation.

> Update warning: do not use the built-in upstream source updater for this modified build. Its update target and the intended Git `origin` remain unresolved. This warning does not fix or disable the updater.

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

The repository has no `pyproject.toml`/`setup.py` declaring `Requires-Python`. The Chinese/Japanese READMEs recommend Python 3.10 or later, and the direct-Python fallback in `run_novel_formatter.command` explicitly rejects versions below 3.10. Release workflows in `.github/workflows/release.yml` use Python 3.12. The local review environment uses Python 3.14.6; that is a tested environment, not a new minimum requirement or proof of compatibility with every supported dependency on Python 3.10.

Use the dependencies declared below and check compatibility for the OCR backend you select; backend environments have separate requirements.

### Python packages

| File | Purpose |
|---|---|
| `requirements.txt` | Core: `PySide6>=6.7,<7`, `Pillow>=10,<13`, `PyMuPDF>=1.24,<2`, `fonttools`, `python-docx`, `pdf2image`, `pillow-heif>=0.18,<2`; AI providers `httpx`, `openai`, `anthropic`, `google-generativeai`; matching/tokens `diff-match-patch`, `rapidfuzz`, `tiktoken`; `epubcheck>=5.3,<6` for EPUB validation |
| `requirements-handwriting-openvino.txt` | Optional independent handwriting recognizer: `numpy>=1.26,<3`, `openvino>=2024.0,<2027` |

HEIC/HEIF support (`pillow-heif`) is part of the core requirements, not an optional extra.

### macOS

Run these commands from the root of an existing copy of this modified source tree. Cloning upstream or downloading an upstream release does not install these local modifications. Create `.venv` only if it does not already exist; reuse an existing environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python gui_pyside6.py
```

For an existing environment, launch directly with `.venv/bin/python gui_pyside6.py`. Alternatively, `./run_novel_formatter.command` delegates to an existing app launcher when present; its fallback searches system Python installations before `.venv`, so it does not guarantee use of the local environment.

**Apple Vision helper.** The native Apple Vision OCR path uses a small Swift helper built by `build_apple_vision_helper.command` from `tools/apple_vision_helper/AppleVisionOCRHelper.swift`. It needs `swiftc` from Xcode or the Command Line Tools (`xcode-select --install`), targets macOS 13.0 by default (`MACOSX_DEPLOYMENT_TARGET`) and writes the binary to `~/Library/Caches/NovelFormatter/apple_vision_helper/<arch>/apple_vision_helper` (override with `NOVEL_FORMATTER_APPLE_VISION_HELPER_DIR`). The binary is ad-hoc signed with `codesign` when available.

```bash
./build_apple_vision_helper.command
```

Apple Vision, the Swift OCR helper and Apple Pencil handwriting recognition are macOS-only.

### Windows

From the root of an existing copy of this modified source tree, create `.venv` only if absent, then install the declared dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python gui_pyside6.py
```

Windows Snipping OCR requires compatible Windows 11 system OCR components. Prebuilt upstream installers are available from [Releases](https://github.com/Amster-Ilvil/Novel-formatter/releases), but do not include these local modifications.

### Optional: RAR/CBR archives

RAR support is optional. It needs the `rarfile` Python package **and** an `unrar` or `bsdtar` binary on `PATH`:

```bash
python -m pip install rarfile
brew install libarchive      # provides bsdtar on macOS; or install unrar
```

Without them, a RAR/CBR input fails with ``RAR support requires the optional `rarfile` package and an unrar/bsdtar binary``; all other formats keep working. Ensure the extraction executable is actually on `PATH` after installation.

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

`run.py` runs image folder → OCR → Formatter → EPUB 3 in one shot. Its OCR path calls Apple Vision through macOS Shortcuts (`--shortcut`, default `ExtractText`), so image OCR from this CLI is macOS-only. `--from-json` and `--from-epub` skip OCR; `--text-layer` reads an existing PDF text layer instead of invoking image OCR. These CLI modes are distinct from the GUI's broader choice of OCR engines.

```bash
# image folder → EPUB
python run.py /path/to/images output.epub --title "Title" --author "Author"

# Page Manager page types, keep the intermediate JSON
python run.py /path/to/images output.epub --overrides overrides.json --save-json intermediate.json

# rebuild from an existing JSON without OCR, left-to-right layout
python run.py --from-json intermediate.json output.epub --ltr

# disable automatic model downloads; use a specific models folder
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
- **Folders** are scanned one level deep in natural order (`p2` before `p10`), expanding images, PDFs and image archives found inside; hidden entries and AppleDouble `._*` files are skipped silently. Sources are deduplicated by `Path.resolve()` before expansion, preserving their first occurrence even when a folder and one of its files are both selected. Distinct source files and separate book pages are not deduplicated by content.
- **Archives:** entries are sorted naturally by their full member path and written as `00001_<stem><ext>`. Directories, `__MACOSX/` and hidden `.`/`._` entries are ignored silently. Unsafe entries (`..` or absolute paths — zip-slip — symlinks, RAR redirections, encrypted entries) and non-image members are reported as skipped; nested archives are not expanded.
- **Limits:** before extraction, image archives are checked against 20 000 entries (including directories) and 4 GiB of declared uncompressed data. For ZIP and non-solid RAR with available, reliable compressed sizes, a member or the combined sized entries exceeding 10 MiB uncompressed and a 100:1 compression ratio is rejected; a zero compressed size also triggers this check above the threshold. The ratio check is not applied to solid RAR or RAR backends without reliable compressed sizes. A separate 4 GiB budget is enforced during streamed image writes, even when metadata understates the size. An archive without any usable image is an error.
- **Cleanup:** cancellation, exceeded limits or extraction I/O errors remove pages and temporary files created by the current extraction. Existing files in an explicit `work_dir` are not overwritten or deleted. Normal imports use application-owned session caches for PDF and archive pages; source-based cache release uses manifests rather than filename extensions, and session shutdown cleans the session root.

## Models: automatic download

Registry-managed 48px and OpenVINO handwriting weights use `utils/model_manager.py`: locate an installed copy → verify → otherwise download → verify → atomically install → use. This is not a universal download path for every OCR backend; see the exceptions below.

### Registry

Data from `MODEL_REGISTRY` in `utils/model_manager.py`:

| Model id | Engine | File | Size (bytes) | SHA-256 | SHA-384 | `hash_source` |
|---|---|---|---:|:-:|:-:|---|
| `manga_48px.checkpoint` | `manga_48px` | `ocr_ar_48px.ckpt` | 204,290,192 | ✓ | — | `existing-code` |
| `manga_48px.dictionary` | `manga_48px` | `alphabet-all-v7.txt` | 186,651 | ✓ | — | `existing-code` |
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

- On first use of an engine the missing files are downloaded with a progress dialog and a **Cancel** button. Verification and downloads run off the GUI thread; progress, failures and cancellation return through the existing Qt signals. Errors have a localized summary with technical details preserved verbatim.
- The manager's default transport requires HTTPS and checks every redirect before requesting its destination. HTTPS-to-HTTP downgrades, other schemes and embedded credentials are rejected. Plain HTTP is limited to numeric loopback addresses for local tests; even loopback is not allowed as an HTTPS downgrade target. An explicitly injected `opener` is a trusted transport and must enforce its own redirect policy. The 48px system-curl fallback requires HTTPS for initial URLs and redirects (`--proto =https --proto-redir =https`), rejects embedded usernames, disables URL globbing, and ignores `.curlrc` so it cannot override these options or disable certificate verification. No additional CA trust or insecure TLS option is supplied.
- Data goes to `<file>.part` in bounded blocks (up to 1 MiB by default) with timeouts, retries with backoff and mirror fallback. Interrupted Python-transport downloads resume via HTTP `Range`; a server ignoring `Range` causes a clean restart, while an invalid `Content-Range` is rejected before appending. Both transports use the same writer, which bounds actual writes, including resumed bytes, by `spec.size`, or by `ModelManager(max_download_bytes=...)` when the size is unknown (default: 4 GiB). `Content-Length` is not the only size check. Exceeding the limit removes the unusable partial file without replacing an existing model.
- After download the size and SHA-256 (plus SHA-384 where pinned) are checked, then the file is moved into place with an atomic `os.replace`. A `.verified` marker keyed on size, mtime and digest avoids re-hashing large checkpoints on every start. This marker is an optimization for a trusted local cache, not a security boundary.
- For a full file hash check without downloading, use `manager.verify(model_id, force=True, target_dir=None)`. It bypasses `.verified` and returns `ModelStatus.READY`, `MISSING` or `CORRUPT`; `force=True` is the default. `force=False` permits the marker optimization. Forced verification of extracted archives raises `ModelError`: the original archive and per-file hashes are not retained, so a full recheck cannot be claimed.
- Concurrent manager callers are serialised per model with a thread lock plus an OS-held file lock (`fcntl.flock` on POSIX, `msvcrt.locking` on Windows). The lock file intentionally persists and must not be deleted to resolve contention. Closing its descriptor or terminating its owner releases the lock; inter-process waiting supports cancellation and `lock_timeout`. The manager checks the cache again after acquisition so a second process reuses the completed download. `stale_lock_seconds` remains accepted for compatibility but no longer expires an active lock.
- The 48px runtime opts into `manager.ensure_spec(..., curl_fallback=True)`. After Python network/HTTP attempts fail, system curl runs inside the same thread/process lock: no lock release or recursive acquisition occurs between transports. Cache recheck, byte limits, size/hash verification and atomic installation all remain manager-owned. Curl writes only to stdout; the manager streams those bytes through its bounded writer without buffering the entire model. An existing destination is retained until a replacement verifies.
- Curl fallback deliberately restarts from zero, discarding the previous `.part` under the lock; it does not use `--continue-at` or curl's internal retries. Ordinary curl transport/HTTP failures may try the next configured HTTPS origin once. URL-policy rejection, integrity failure, cancellation, byte-limit rejection and curl timeout are terminal, not reasons to try another transport or origin. A failed curl attempt leaves no partial download for a later resume. Python's existing network-interruption resume remains available when curl fallback is not enabled.
- Curl runs with an argument list and no shell. Stdout buffering is bounded; stderr is drained concurrently and only its last 8 KiB is retained. Connection/inactivity timeout uses `ModelManager.timeout` (30 seconds by default), with a separate 1,800-second total deadline. Cancellation, timeout and write/verification failures stop and reap the subprocess, close streams and release the download lock; termination escalates to kill after one second. Missing/unstartable curl and nonzero exits report the fallback failure while preserving the original Python failure as the exception cause. Local-file imports and download-disabled checks remain available without starting curl.
- Transport regressions use controlled local subprocesses and event-synchronised competing processes, not real model downloads. Optional real-curl tests require curl and OpenSSL: a local HTTPS fixture checks successful HTTPS redirects, certificate rejection, blocked HTTPS-to-HTTP and credential-bearing redirects, and streaming overflow. The fixture passes its own certificate only to the test subprocess; it does not modify system trust. These tests do not certify every curl TLS backend or platform.
- Errors are raised as `ModelError` subclasses with model context:
  - `ModelNetworkError` — no network, DNS failure, refused connection or timeout; timeout messages identify the model and applied request timeout separately from offline failures, while retaining manual-installation guidance;
  - `ModelDownloadError` — an HTTP error status or a rejected URL/redirect;
  - `ModelIntegrityError` — size, hash, resume range, archive layout or archive safety check failed (the unusable download is discarded);
  - `ModelCancelledError` — cancelled by the user;
  - `ModelNotAvailableError` — the model is missing and downloading is not allowed.

### Offline use and manual installation

- **Settings → OCR → Download models automatically when needed** is on by default. Turning it off, passing `run.py --no-download` or setting `NOVEL_FORMATTER_MODEL_AUTO_DOWNLOAD=0` (also `false`, `no`, `off`) disables automatic downloads; a missing model then fails with `ModelNotAvailableError` naming the download URL and target folder. Explicit "Download OCR Models" buttons still work when automatic downloads are off.
- **Manual installation:** download the files from the URLs listed in `MODEL_REGISTRY` and place them under the models folder, keeping the file names from the table, e.g. `<models folder>/manga-48px-ar/ocr_ar_48px.ckpt` and `<models folder>/handwriting_openvino/handwritten-japanese-recognition-0001.xml`. They are verified against the pinned hashes on next use.
- Download preference precedence is explicit application configuration, then `NOVEL_FORMATTER_MODEL_AUTO_DOWNLOAD`, then the default of enabled. A saved GUI setting can therefore override the environment variable.
- These controls disable automatic model downloads in the paths that honor the setting; they are not an application-wide network sandbox. Prepare backend runtimes and weights before working offline. Explicit model-install/update actions and remote AI services are separate operations.

## Left-to-right (horizontal) conversion

An optional export step (`builder/text_direction.py`) turns a vertical Japanese EPUB into a horizontal left-to-right publication.

**Where to enable it:**

- GUI: open the EPUB workspace, use its advanced build view, then enable **Convert to left-to-right (horizontal) layout**. Returning to the compact view preserves the checkbox; its reading-direction selector reflects the effective horizontal output. Explicitly selecting vertical output in the compact view clears the LTR override. The choice is session-scoped, not a saved application preference.
- CLI: `run.py ... --ltr` or `python -m builder.epub_builder ... --ltr`;
- API: `build_epub(..., text_direction="ltr")`; the default is `"keep"` (`TextDirectionMode.KEEP`).

**What it does:**

- sets `page-progression-direction="ltr"` on the spine, `dir="ltr"` on documents and `writing-mode: horizontal-tb`;
- removes vertical-only CSS rules;
- replaces supported vertical presentation-form punctuation in the U+FE10–FE19 and U+FE30–FE4F ranges with horizontal forms in XHTML text, without rewriting attributes, scripts or comments;
- unwraps purely presentational tate-chū-yoko (horizontal-in-vertical) spans while keeping ruby and spans carrying other semantics such as an `id`;
- for a vertical source, swaps the halves of detected two-page illustration spreads so the left half comes first; this depends on automatic spread detection, not a general page-order reconstruction.

**What it does not do:**

- it does not reverse chapter order or the general page sequence; detected illustration pairs are the specific exception above;
- it does not re-run OCR or the Formatter, translate the book, or fix recognition errors.

**Regression coverage and limits:**

- Explicit `KEEP` and the default path produce equivalent publication content in regression fixtures after normalizing generated UUIDs and modification timestamps. This is not a promise of byte-identical EPUB ZIP files across separate builds.
- An already-horizontal regression fixture without vertical presentation forms only gains explicit `dir="ltr"` in XHTML; its CSS and spine already match horizontal output. Other input can still need punctuation or markup conversion. The conversion functions have idempotence tests.
- Offscreen UI regressions exercise compact/advanced mode changes, retained LTR selection, signal connections and real EPUB export with LTR both enabled and disabled.
- `--ltr` is rejected in `run.py --batch` mode.

## Interface language

The interface defaults to English (`en_US`) on a fresh install. A previously saved Simplified Chinese (`zh_CN`) or Japanese (`ja_JP`) choice is kept. These are the three implemented UI languages. Switch between English / 简体中文 / 日本語 in Settings → Interface & Appearance → Interface Language; the change applies without restarting and affects UI chrome rather than translating OCR text, documents or model identifiers. `run.py` argument help is English; this is not a claim that every backend log or exception is English.

Russian UI is not implemented: there is no Russian entry in `SUPPORTED_LANGUAGES` or Russian translation catalog. The original requirement for a Russian source UI remains unmet; this review does not add a new language.

Dynamic page headers, including the lazily created EPUB workspace, retain their source strings for repeated translation. Offscreen tests check navigation and language switching, the direction indicator and the Clear EPUB button. A zero-untranslated static audit alone does not prove that all dynamically constructed UI is translated.

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

New scoped measurements (2026-11-06) compare verified baseline `b962ff0` with
HEAD `f6d550b` plus the existing working tree on Apple M3 Pro / macOS 27.2.
The [measurement report](benchmarks/results/20261106T162342Z-aox7uez3/report.md) records exact source
and input hashes, local runtime/model versions, raw samples, profiles and commands.

Each cell below is seconds: median (min–max), with five measured repetitions
per variant and phase. Cold means the first operation in a fresh process,
not a cleared OS file cache; parent imports are timed separately. Warm follows
one excluded warmup after the cold call. Baseline/current is the ratio of medians.

| Scenario | Phase | n per variant | Baseline seconds: median (min–max) | Current seconds: median (min–max) | Baseline/current |
|---|---|---:|---:|---:|---:|
| Connected components | cold | 5 | 0.020708 (0.019108–0.021403) | 0.009578 (0.009061–0.010271) | 2.162 |
| Connected components | warm | 5 | 0.018725 (0.017574–0.025932) | 0.009039 (0.008873–0.011038) | 2.072 |
| Column detection | cold | 5 | 0.073933 (0.072351–0.077551) | 0.065534 (0.061274–0.071743) | 1.128 |
| Column detection | warm | 5 | 0.071179 (0.069762–0.089309) | 0.060521 (0.059314–0.066010) | 1.176 |
| Synthetic PNG-to-document | cold | 5 | 0.261814 (0.241419–0.304898) | 0.261081 (0.231795–0.271781) | 1.003 |
| Synthetic PNG-to-document | warm | 5 | 0.216399 (0.201295–0.242013) | 0.193846 (0.165893–0.206571) | 1.116 |
| Real NDLOCR session + page | cold | 5 | 3.451358 (3.234411–4.295713) | 3.326603 (3.131470–3.651849) | 1.038 |
| Real NDLOCR session + page | warm | 5 | 3.399563 (3.119896–3.469095) | 0.499498 (0.483474–0.586932) | 6.806 |

Real OCR used the already-installed NDLOCR-Lite models on CPU, offline, on one
1167 × 1653 text-page PNG. The warm session lifecycle plus full-page request
improved from 3.400 s to 0.499 s (6.806 ratio), with matching complete OCR block
outputs and worker-PID evidence of reuse. This is not an inference-kernel-only
or PDF-to-EPUB speedup. Cold real OCR has overlapping ranges; no reliable cold
speedup is claimed.

The algorithm cases use a deterministic 1240 × 1754 fixture: 922 components,
37 detected columns, and matching historical references for 694 column images.
The synthetic case uses fake OCR and measures PNG-to-document processing only;
it bypasses the session cache. Its cold result is effectively unchanged, and
its warm ranges overlap, so no reliable synthetic-pipeline speedup is claimed.
The synthetic comparison also includes the existing PNG input-handling path
in the modified `pdf_input.py`; it is not an isolated segmentation-only metric.

Reproduce with the existing local runtime/models, from the project root:

```sh
sh benchmarks/results/20261106T162342Z-aox7uez3/reproduce.sh
```

The reproduction script creates a detached baseline worktree and unique result
directory, without switching the current branch or copying uncommitted files.
The new `tools/measure_ocr_optimizations.py` refuses an existing `--out` directory.
Raw cold, warmup and warm samples, min/max, MAD, output signatures and text
cProfile tables are retained. No models, large binary profiles or personal
absolute paths are included.

The older `tools/bench_pipeline.py` remains available for the 42-page synthetic
PDF pipeline (40 text pages, cover, illustration; default 200 DPI). It replaces
model inference with a fake recognizer while exercising rendering, column
processing, document assembly, formatting and EPUB creation. That full pipeline
was not remeasured here, and its earlier unarchived numerical claims remain
unverified. Full real-OCR PDF-to-EPUB speed and recognition quality are unmeasured.

Warning: the old script deletes and recreates an existing `--out-dir`.
Never point it at an existing user directory. If using it separately, allocate
a new temporary parent and a previously nonexistent child output path for each
checkout; retain both reports and input/output signatures.


## Development and tests

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
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
