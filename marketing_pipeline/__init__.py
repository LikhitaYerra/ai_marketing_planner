"""Marketing pipeline package providing orchestrated content planning utilities."""

from .config import FeatureConfig, load_config
from .models import RunInput, RunResult
from .orchestrator import run_pipeline

__all__ = [
    "FeatureConfig",
    "load_config",
    "RunInput",
    "RunResult",
    "run_pipeline",
]
