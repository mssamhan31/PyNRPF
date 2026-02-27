from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd


class BaseModelPlugin(ABC):
    name: str

    @abstractmethod
    def run_inference(
        self,
        df: pd.DataFrame,
        cfg: Dict[str, Any],
        columns: Dict[str, str],
    ) -> pd.DataFrame:
        raise NotImplementedError

    def train(
        self,
        df: pd.DataFrame,
        cfg: Dict[str, Any],
        columns: Dict[str, str],
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError(f"Training is not implemented for model '{self.name}'.")
