#!/usr/bin/env python3
"""Pre-download the RapidOCR model for mixed Russian + English (Cyrillic) OCR.

RapidOCR ships a dedicated East-Slavic recognition model (``eslav_PP-OCRv5_rec``)
whose character dictionary covers the full Cyrillic alphabet *and* the Latin
alphabet + digits + punctuation. A single recognition model therefore handles
documents that mix Russian and English text (even within the same line); no
second OCR engine is required.

This script downloads the East-Slavic ONNX recognition model and its character
dictionary into a target directory, then prints a ready-to-use docling-serve OCR
preset that points ``rec_model_path`` / ``rec_keys_path`` at those files. The
detection and classification models are script-agnostic and are fetched
separately by ``docling-tools models download rapidocr`` (or downloaded by
RapidOCR on first use), so only the language-specific recognition model is
handled here.

Note: the recognition language *cannot* be selected through ``rapidocr_params``
(e.g. ``Rec.lang_type``): docling passes those values straight to RapidOCR, which
requires Enum types there and rejects plain strings. Using explicit model paths
is the reliable, JSON-friendly way to wire in the East-Slavic model.

Environment variables:
    RAPIDOCR_LANG     Recognition model language. "eslav" (PP-OCRv5, default);
                      covers Cyrillic + Latin.
    RAPIDOCR_OUT_DIR  Target directory for the downloaded files.
                      Default: ./rapidocr_ru_en_models
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

_BASE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {url}\n    -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        dest.write_bytes(resp.read())


def main() -> int:
    lang = os.environ.get("RAPIDOCR_LANG", "eslav").strip().lower()
    out_dir = Path(
        os.environ.get("RAPIDOCR_OUT_DIR", "rapidocr_ru_en_models")
    ).expanduser()

    rec_name = f"{lang}_PP-OCRv5_rec_mobile.onnx"
    dict_name = f"ppocrv5_{lang}_dict.txt"
    rec_path = out_dir / rec_name
    dict_path = out_dir / dict_name

    print(f"Downloading RapidOCR {lang} PP-OCRv5 recognition model + dictionary...")
    try:
        _download(f"{_BASE}/onnx/PP-OCRv5/rec/{rec_name}", rec_path)
        _download(
            f"{_BASE}/paddle/PP-OCRv5/rec/{lang}_PP-OCRv5_rec_mobile/{dict_name}",
            dict_path,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clear message to the user
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    preset = {
        "rapidocr_ru_en": {
            "kind": "rapidocr",
            "backend": "onnxruntime",
            "lang": ["english"],
            "rec_model_path": str(rec_path.resolve()),
            "rec_keys_path": str(dict_path.resolve()),
        }
    }
    print(
        "\nDone. Configure docling-serve with:\n"
        f"  export DOCLING_SERVE_CUSTOM_OCR_PRESETS='{json.dumps(preset)}'\n"
        'Then request conversions with "ocr_preset": "rapidocr_ru_en".'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
