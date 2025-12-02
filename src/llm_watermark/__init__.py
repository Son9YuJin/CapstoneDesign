"""
LLM text watermarking library.
"""

from .processor import WatermarkLogitsProcessor, WatermarkDetector

try:
    from .extended_processor import (
        ExtendedWatermarkLogitsProcessor,
        ExtendedWatermarkDetector,
    )
except ImportError:  
    ExtendedWatermarkLogitsProcessor = None
    ExtendedWatermarkDetector = None

__all__ = [
    "WatermarkLogitsProcessor",
    "WatermarkDetector",
    "ExtendedWatermarkLogitsProcessor",
    "ExtendedWatermarkDetector",
]
