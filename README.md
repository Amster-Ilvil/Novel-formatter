# Novel Formatter

Novel Formatter is a macOS desktop application for OCR, text comparison,
formatting, EPUB export, and Japanese image-text review.

## Run

Requirements:

- macOS 15 or newer
- Python 3.10 or newer
- PySide6 and the packages listed in `requirements.txt`

Install the main runtime dependencies and start the GUI:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python gui_pyside6.py
```

On macOS, `run_novel_formatter.command` is the preferred launcher. The app
shell can prepare its own Python runtime and apply source updates from the
configured update directory. OCR model files are downloaded only when the
corresponding OCR feature is explicitly selected; model files and caches are
kept outside this repository.

## Included

This repository contains the application source, runtime assets, native helper
source, third-party notices required by the handwriting review feature, and
startup/update scripts. It does not contain model weights, virtual
environments, user documents, OCR results, logs, API keys, or local settings.

## Privacy

The application does not include credentials or personal data in this
repository. AI provider credentials are entered by the user at runtime and
are stored in the local application settings. Input documents and generated
outputs remain local unless the user explicitly chooses an external OCR or AI
provider.

## License

See the included third-party notices before redistributing bundled assets.
