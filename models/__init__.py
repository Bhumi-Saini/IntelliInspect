"""Models package for IntelliInspect."""
from .predict import DefectDetector, get_detector
from .evaluate import evaluate_model, run_model_evaluation

__all__ = [
    "DefectDetector",
    "get_detector",
    "evaluate_model",
    "run_model_evaluation"
]
