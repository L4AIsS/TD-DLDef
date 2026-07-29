"""TD-DLDef revised reference implementation.

The package implements constraint-aware model generation, diversity-guided layer
selection, six model mutation operators, contract-aware test oracles, and a
reproducible experiment harness.
"""

from .version import __version__
from .model_ir import LayerCandidate, LayerNode, ModelGraph, TensorSpec
from .diversity import DiversityGain, DiversityTracker
from .bandits import create_policy

__all__ = [
    "__version__",
    "TensorSpec",
    "LayerNode",
    "LayerCandidate",
    "ModelGraph",
    "DiversityGain",
    "DiversityTracker",
    "create_policy",
]
