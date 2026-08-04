# Manga Image Translator 48px AR OCR notice

This project adds an optional runtime adapter for the `ocr_ar_48px.ckpt` model
published by `zyddnys/manga-image-translator`.

The checkpoint, alphabet and upstream recognizer source are **not bundled** in
this archive. On first use, the adapter downloads them from the official GitHub
repository/release and verifies the published SHA-256 values for the checkpoint
and alphabet. Cached upstream source keeps its original notices. The XPOS helper
is Copyright (c) 2022 Microsoft and licensed under the MIT License.

Upstream project: `https://github.com/zyddnys/manga-image-translator`
