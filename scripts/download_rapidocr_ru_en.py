#!/usr/bin/env python3
"""Pre-download RapidOCR models for mixed Russian + English (Cyrillic) OCR.

RapidOCR ships a dedicated East-Slavic recognition model (``eslav_PP-OCRv5_rec``)
whose character dictionary covers the full Cyrillic alphabet *and* the Latin
alphabet + digits + punctuation. A single recognition model therefore handles
documents that mix Russian and English text (even within the same line); no
second OCR engine is required.

Running this script instantiates a ``RapidOCR`` reader with the desired language
and backend, which triggers the (one-time) download of the detection,
classification and recognition models into RapidOCR's local cache. Run it during
image build / deployment so the runtime works without outbound network access.

Environment variables:
    RAPIDOCR_LANG     Recognition language. "eslav" (PP-OCRv5, default) or
                      "cyrillic" (also covers Cyrillic + Latin).
    RAPIDOCR_BACKEND  Inference backend: "onnxruntime" (default) or "paddle".
                      "onnxruntime" ships with the ``rapidocr`` extra; "paddle"
                      additionally requires the ``paddlepaddle`` package.
"""

from __future__ import annotations

import os
import sys

# Recognition model version per language (eslav is only published for PP-OCRv5).
_OCR_VERSION_BY_LANG = {
    "eslav": "PP-OCRv5",
    "cyrillic": "PP-OCRv5",
}


def main() -> int:
    lang = os.environ.get("RAPIDOCR_LANG", "eslav").strip().lower()
    backend = os.environ.get("RAPIDOCR_BACKEND", "onnxruntime").strip().lower()
    ocr_version = _OCR_VERSION_BY_LANG.get(lang, "PP-OCRv5")

    try:
        from rapidocr import RapidOCR
    except ImportError:
        print(
            "rapidocr is not installed. Install it with:\n"
            "  pip install 'docling-serve[rapidocr]'   # onnxruntime backend\n"
            "(the 'paddle' backend additionally requires the paddlepaddle package)",
            file=sys.stderr,
        )
        return 1

    # Keys match RapidOCR's config.yaml. The recognition stage drives the
    # language; detection/classification stay on their defaults (text detection
    # is script-agnostic). When *_model_path is unset, RapidOCR downloads the
    # model that matches engine_type + lang_type + ocr_version.
    params = {
        "Det.engine_type": backend,
        "Cls.engine_type": backend,
        "Rec.engine_type": backend,
        "Rec.lang_type": lang,
        "Rec.ocr_version": ocr_version,
    }

    print(f"Warming up RapidOCR (lang={lang}, backend={backend}, {ocr_version})...")
    RapidOCR(params=params)
    print(
        "RapidOCR models cached. Configure docling-serve with a custom OCR preset, "
        "e.g.:\n"
        '  DOCLING_SERVE_CUSTOM_OCR_PRESETS=\'{"rapidocr_ru_en": '
        f'{{"kind": "rapidocr", "backend": "{backend}", '
        f'"rapidocr_params": {{"Rec.lang_type": "{lang}", '
        f'"Rec.ocr_version": "{ocr_version}"}}}}}}\''
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
