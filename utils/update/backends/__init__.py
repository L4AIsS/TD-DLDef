"""Backend adapters."""

from .base import Backend, BackendUnavailable
from .numpy_backend import NumpyBackend


def create_backend(name: str, **kwargs):
    key = name.strip().lower()
    if key == "numpy":
        return NumpyBackend()
    if key in {"torch", "pytorch"}:
        from .torch_backend import TorchBackend
        return TorchBackend(**kwargs)
    if key in {"tensorflow", "tf"}:
        from .tensorflow_backend import TensorFlowBackend
        return TensorFlowBackend(**kwargs)
    raise ValueError(f"Unknown backend {name!r}")


__all__ = ["Backend", "BackendUnavailable", "NumpyBackend", "create_backend"]
