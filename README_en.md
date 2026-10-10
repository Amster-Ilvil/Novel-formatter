<div align="center">

[简体中文](README.md) · **English** · [日本語](README_ja.md)

</div>

<p align="center">
  <img src="assets/novel_formatter_banner.webp" alt="Novel Formatter" width="100%">
</p>

<h1 align="center">Novel Formatter</h1>

<p align="center"></p>

<p align="center">
  <img alt="macOS" src="https://img.shields.io/badge/macOS-Apple%20Silicon%20%2F%20Intel-black?logo=apple">
  <img alt="Windows" src="https://img.shields.io/badge/Windows-x64-0078D4?logo=windows11&logoColor=white">
  <img alt="PySide6" src="https://img.shields.io/badge/UI-PySide6-41CD52?logo=qt&logoColor=white">
  <img alt="Japanese OCR" src="https://img.shields.io/badge/OCR-Japanese-2f6feb">
  <img alt="EPUB" src="https://img.shields.io/badge/Output-EPUB-8A2BE2">
</p>

> Japanese vertical OCR · Multi-engine OCR comparison and fusion · Image/text proofreading · Formatter · EPUB · AI review packages · Apple Vision

## Main features

- **Image/PDF import:** Process individual images, image folders, and PDF pages.
- **Vertical Japanese OCR:** Handle vertical writing, individual text columns, and text spanning columns in Japanese books.
- **Multi-engine OCR comparison and fusion:** Review results from Apple Vision, Windows Snipping OCR, NDLOCR-Lite, Hayai OCR, 48px OCR, PaddleOCR, and other supported engines.
- **Image/text proofreading:** Compare the source page with its OCR output to find incorrect characters, omissions, low-confidence text, and layout problems.
- **Ruby, headers, and page numbers:** FindText CenterNet helps detect ruby/furigana areas and remove running headers, page numbers, and fragments across columns or pages.
- **Formatter:** Clean up paragraphs, punctuation, headings, and cross-page continuation after OCR.
- **EPUB creation:** Generate EPUB from corrected text, book structure, and embedded resources.
- **AI review packages:** Export evidence, text, structure, and resources for further review with GPT, Gemini, GLM, DeepSeek, and compatible services, helping prepare publication-ready EPUB files.
- **Local data first:** User documents, model weights, caches, logs, and databases are not committed to the source repository.

## What's new in v2.0

Compared with v1.3, v2.0 expands the OCR, proofreading, PDF, and EPUB workflows:

- **New project workspace:** Store pages, OCR results, decisions, formatted text, and EPUB production state in one workspace; restore work and browse recent projects.
- **Multi-engine OCR and adjudication:** Refine per-column and full-page recognition roles, disagreement review, image/text comparison, and manual decisions; improve integration with Hayai OCR, NDLOCR-Lite, and 48px OCR.
- **Native system OCR:** Add Windows 11 Snipping OCR and improve the native Apple Vision path on Intel and Apple Silicon Macs.
- **PDF text layer:** Use original PDFs and physical-page mappings from Page Management for better vertical reading order, cross-page continuation, paragraphs, and publication-oriented cleanup.
- **AI review improvements:** Support GPT, Gemini, GLM, DeepSeek, and compatible providers while strengthening safe round-tripping of packages, OCR evidence, and manual edits.
- **EPUB/TEI and UI:** Improve cover images, illustrations, page boundaries, metadata, EPUB structure, previews, progress reporting, and stability across operating systems.

## OCR engines

Supported or optional:

- Apple Vision (macOS)
- Windows Snipping OCR (Windows 11)
- NDLOCR-Lite
- Hayai OCR
- 48px OCR
- PaddleOCR
- Paddle VL MLX (macOS Apple Silicon)
- FindText CenterNet (ruby-region detection aid)

Apple Vision, Swift OCR Helper, and Apple Pencil handwriting recognition are only available on macOS. Windows Snipping OCR requires compatible Windows 11 system components.

## Recommended workflow

1. Import a PDF, image, or image folder.
2. Label the cover, title page, table of contents, illustrations, body pages, afterword, and other page types.
3. Choose the OCR engines and optionally enable column-by-column recognition.
4. Run OCR.
5. Inspect disagreements, missing text, ruby, page numbers, and cross-page continuation in OCR Comparison and Image/Text Review.
6. Apply the fused result and format the text using Formatter.
7. Export EPUB directly or export an AI review package for more proofreading.

## AI review packages

AI review packages combine OCR evidence, text, structure, and resources, allowing models to check the original context rather than an isolated text transcript.

They can assist with:

- Correcting OCR mistakes, missing text, and punctuation
- Checking names, locations, and terms for consistency
- Reviewing context across pages and chapters
- Checking EPUB structure and typography
- Producing a more publication-ready EPUB

## Installation and startup

Python 3.10 or later is recommended.

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

Alternatively:

```bash
./run_novel_formatter.command
```

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

Prebuilt installers are available from [Releases](https://github.com/Amster-Ilvil/Novel-formatter/releases).

## Platform support

| Feature | macOS | Windows |
|---|---:|---:|
| Main interface | ✓ | ✓ |
| PDF/image OCR | ✓ | ✓ |
| Multi-engine OCR | ✓ | ✓ |
| Formatter/EPUB | ✓ | ✓ |
| AI review packages | ✓ | ✓ |
| Apple Vision | ✓ | — |
| Windows Snipping OCR | — | Windows 11 |
| Swift OCR Helper | ✓ | — |
| Apple Pencil handwriting recognition | ✓ | — |

## Models and local data

OCR model weights, caches, virtual environments, logs, databases, and user output are not committed to the repository. Some OCR engines prepare dependencies or model files locally when first used.

## References and acknowledgments

The UI, OCR integrations, document processing, and handwriting-recognition features use or draw inspiration from the following projects and documentation:

- [Qt for Python / PySide6](https://doc.qt.io/qtforpython/): Desktop GUI framework.
- [Apple Vision](https://developer.apple.com/documentation/vision): macOS-native OCR and vision APIs.
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): General OCR and cross-platform reference.
- [NDLOCR-Lite](https://github.com/ndl-lab/ndlocr-lite): OCR and layout analysis for Japanese books.
- [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo): Optional handwriting models and inference examples.
- [PyMuPDF](https://pymupdf.readthedocs.io/): PDF pages, images, and text layers.
- [python-docx](https://python-docx.readthedocs.io/): Read and generate DOCX.
- [Pillow](https://python-pillow.org/): Image loading, cropping, and preprocessing.
- [jlect-jhr](https://github.com/ZacharyRead/jlect-jhr): Japanese handwriting resources; see `third_party/jlect_jhr/LICENSE.txt`.

Third-party projects, models, and resources have their own licenses. No OCR model weights are redistributed in this repository.

## Star History

<p align="center">
  <a href="https://www.star-history.com/?type=date&repos=Amster-Ilvil%2FNovel-formatter">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Amster-Ilvil/Novel-formatter&type=Date&theme=dark">
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Amster-Ilvil/Novel-formatter&type=Date">
      <img alt="Novel Formatter Star History Chart" src="https://api.star-history.com/svg?repos=Amster-Ilvil/Novel-formatter&type=Date" width="800">
    </picture>
  </a>
</p>


## Use and license

The project's original code and documentation are made available under the [MIT License](LICENSE) for personal, educational, research, and commercial use, subject to its terms.

Third-party OCR engines, models, dependencies, and platform capabilities remain governed by their respective licenses and terms. See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for full credits.
