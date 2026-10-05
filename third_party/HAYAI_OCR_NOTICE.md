# Hayai OCR third-party notice

Novel Formatter can optionally install and use **Hayai OCR v2.5-nova** on demand.
No Hayai package files or model weights are bundled in this source archive.

- Python project: `NopeNopeGuy/hayai-ocr`
- Pinned package contract: `hayai-ocr==2.3.0`
- Torch model: `JustANormalTinkerer/hayai-ocr-v2`
- LiteRT model: `JustANormalTinkerer/hayai-ocr-v2-tflite`
- Vision processor used by upstream v2: `google/siglip2-base-patch16-naflex`
- Upstream project/model license: Apache License 2.0

The application creates a dedicated `.venv-hayai-ocr` and a dedicated
`.model-cache/hayai-ocr` only after the user explicitly confirms first use.
After a successful local inference, normal OCR starts reuse the verified cache
offline so an upstream repository change cannot silently replace the working
snapshot. Removing the dedicated cache/runtime state intentionally re-enables a
fresh first-use download.

See the upstream repositories for complete copyright and license texts.
