# Windows Snipping OCR

`Windows Snipping OCR` is the Windows 10/11 system-OCR backend requested in issue #11.

## What it uses

Novel Formatter discovers the installed **Microsoft Snipping Tool** package (`Microsoft.ScreenSketch`) and, when available, uses the OneOCR runtime shipped in its `SnippingTool` directory:

- `oneocr.dll`
- `oneocr.onemodel`
- `onnxruntime.dll`

These files are **not** copied into this repository, downloaded by Novel Formatter, or redistributed in release packages. The adapter loads the user's already-installed system package in place.

The integration was independently implemented after reviewing the public behavior documented by the LunaTranslator and community OneOCR integrations. No LunaTranslator source code is vendored into Novel Formatter.

## Availability

The engine is shown only on Windows. At run time Novel Formatter:

1. checks `NOVEL_FORMATTER_SNIPPING_OCR_DIR` when explicitly set;
2. otherwise queries the installed `Microsoft.ScreenSketch` AppX package;
3. requires all three OneOCR assets above;
4. fails with an actionable error if the package is missing or incomplete.

Updating **Snipping Tool** from Microsoft Store is the normal repair path.

## OCR pipeline integration

The engine supports both normal whole-page OCR and Novel Formatter's existing Japanese physical-column path. It is also a valid multi-model role/review engine, so it can participate in the same geometry, comparison and adjudication workflow as other local OCR engines.

OneOCR line boxes and word confidences are converted to Novel Formatter's normal block protocol. When a Snipping Tool build does not expose word confidence, the adapter reports confidence as unavailable rather than fabricating a score.

## Compatibility boundary

OneOCR is an implementation detail of Snipping Tool, not a documented public Microsoft OCR API. Its binary export ABI can change in a future Windows/Snipping Tool release. For that reason:

- the integration lives in an isolated worker process;
- required exports are validated before recognition;
- ABI failures are reported explicitly;
- no fallback silently substitutes a different OCR engine.

This keeps a future system update from corrupting OCR evidence or destabilizing the main GUI process.

## Manual diagnostics

On Windows:

```powershell
python adapters/windows_snipping_ocr_adapter.py
```

The command prints whether the local Snipping Tool OneOCR runtime can be discovered. An advanced portable/package layout may override discovery:

```powershell
$env:NOVEL_FORMATTER_SNIPPING_OCR_DIR = 'C:\path\to\SnippingTool'
```

The override must still contain the genuine OneOCR runtime files listed above.
