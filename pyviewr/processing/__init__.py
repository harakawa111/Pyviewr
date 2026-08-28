"""Image processing (enhance + UV sunscreen detection)."""

from pyviewr.processing.enhance import EnhanceParams, apply as apply_enhance
from pyviewr.processing.sunscreen import DetectionParams, process_frame

__all__ = [
    "EnhanceParams",
    "apply_enhance",
    "DetectionParams",
    "process_frame",
]
