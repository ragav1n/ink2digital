"""
Model implementations for German Lecture Slide OCR.

Modules:
    dlaformer_adapter   - DLAFormer document layout detector (Phase 2)
    math_ocr_tamer      - TAMER math expression recognizer (Phase 2)
    meta_learning_ocr   - MAML OCR wrapper for professor adaptation (Phase 3)
    llm_corrector       - Local LLM (Qwen3-8B) German OCR-text corrector (Phase 4)
"""

try:
    from .llm_corrector import LLMCorrector  # noqa: F401
except ImportError:
    # The LLM corrector and its deps (editdistance, etc.) are not used on the
    # VLM / MLX inference path. Skip it rather than break `import models.*`
    # when those optional deps are absent (e.g. the Mac inference install).
    pass
