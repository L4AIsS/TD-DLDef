"""Backend protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np

from ..execution import ExecutionResult
from ..model_ir import ModelGraph


class BackendUnavailable(RuntimeError):
    pass


class Backend(ABC):
    name = "base"

    @property
    @abstractmethod
    def version(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def execute(
        self,
        graph: ModelGraph,
        inputs: Mapping[str, np.ndarray],
        *,
        contract_valid: bool = True,
        expected_rejection: bool = False,
    ) -> ExecutionResult:
        raise NotImplementedError


    def input_gradient(
        self,
        graph: ModelGraph,
        inputs: Mapping[str, np.ndarray],
        *,
        epsilon: float = 1e-4,
        max_elements: int = 256,
    ) -> ExecutionResult:
        raise NotImplementedError(f"{self.name} does not implement input gradients")

    def environment(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version}
