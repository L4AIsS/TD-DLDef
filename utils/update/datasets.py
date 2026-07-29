"""Dataset loaders matching the paper's MNIST, CIFAR-10, and SineWave setup.

Network downloads are never performed silently by the core artifact. MNIST/CIFAR-10
use a local NPZ when supplied, otherwise an installed Keras loader (which may download
according to Keras' own cache policy). SineWave is generated deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import TaskSpec, built_in_task
from .execution import numpy_dtype
from .model_ir import TensorSpec


@dataclass(slots=True)
class DatasetBundle:
    name: str
    x: np.ndarray
    y: np.ndarray | None
    task: TaskSpec
    split: str
    metadata: dict[str, Any]

    def sample(self, rng: np.random.Generator, spec: TensorSpec) -> dict[str, np.ndarray]:
        index = int(rng.integers(0, len(self.x)))
        value = np.asarray(self.x[index : index + 1])
        if tuple(value.shape) != tuple(spec.shape):
            raise ValueError(f"Dataset sample shape {value.shape} does not match model input {spec.shape}")
        return {"input": value.astype(numpy_dtype(spec.dtype), copy=False)}


def _keras_datasets():
    try:
        from keras import datasets
        return datasets
    except Exception:
        try:
            from tensorflow.keras import datasets
            return datasets
        except Exception as exc:
            raise RuntimeError(
                "MNIST/CIFAR-10 loader requires Keras/TensorFlow or a local NPZ file"
            ) from exc


def _limit(x, y, limit: int | None):
    if limit is None or limit <= 0:
        return x, y
    return x[:limit], None if y is None else y[:limit]


def load_dataset(
    name: str,
    *,
    split: str = "test",
    local_path: str | Path | None = None,
    limit: int | None = None,
    seed: int = 0,
    sine_length: int = 50,
    sine_samples: int = 2048,
) -> DatasetBundle:
    key = name.strip().lower()
    split = split.lower()
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")

    if key in {"synthetic", "random"}:
        task = built_in_task("vision")
        rng = np.random.default_rng(seed)
        count = limit or 256
        x = rng.normal(size=(count,) + task.input_spec.shape[1:]).astype(np.float32)
        return DatasetBundle("synthetic", x, None, task, split, {"seed": seed})

    if key == "mnist":
        if local_path:
            raw = np.load(local_path)
            x = raw[f"x_{split}"]
            y = raw.get(f"y_{split}")
        else:
            (x_train, y_train), (x_test, y_test) = _keras_datasets().mnist.load_data()
            x, y = (x_train, y_train) if split == "train" else (x_test, y_test)
        x = np.asarray(x, dtype=np.float32) / 255.0
        if x.ndim == 3:
            x = x[..., None]
        x, y = _limit(x, y, limit)
        task = TaskSpec("vision", TensorSpec((1, 28, 28, 1), "float32", "image"), "dataset")
        return DatasetBundle("mnist", x, y, task, split, {"normalization": "x/255"})

    if key in {"cifar10", "cifar-10"}:
        if local_path:
            raw = np.load(local_path)
            x = raw[f"x_{split}"]
            y = raw.get(f"y_{split}")
        else:
            (x_train, y_train), (x_test, y_test) = _keras_datasets().cifar10.load_data()
            x, y = (x_train, y_train) if split == "train" else (x_test, y_test)
        x = np.asarray(x, dtype=np.float32) / 255.0
        x, y = _limit(x, y, limit)
        task = TaskSpec("vision", TensorSpec((1, 32, 32, 3), "float32", "image"), "dataset")
        return DatasetBundle("cifar10", x, y, task, split, {"normalization": "x/255"})

    if key in {"sine", "sinewave", "sine-wave"}:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=sine_samples)
        frequencies = rng.uniform(0.5, 2.0, size=sine_samples)
        t = np.linspace(0.0, 2.0 * np.pi, sine_length + 1)
        waves = np.sin(phases[:, None] + frequencies[:, None] * t[None, :]).astype(np.float32)
        x = waves[:, :-1, None]
        y = waves[:, 1:, None]
        x, y = _limit(x, y, limit)
        task = TaskSpec("transformer", TensorSpec((1, sine_length, 1), "float32", "sequence"), "dataset")
        return DatasetBundle("sinewave", x, y, task, split, {"seed": seed, "length": sine_length})

    raise ValueError(f"Unknown dataset {name!r}")
