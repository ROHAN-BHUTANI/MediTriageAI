from abc import ABC, abstractmethod
from collections.abc import Iterator

import pandas as pd


class BaseAdapter(ABC):
    @property
    @abstractmethod
    def dataset_source(self) -> str:
        """The canonical name of the dataset source."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Adapter version string."""

    @abstractmethod
    def ingest(self, raw_path: str) -> Iterator[pd.DataFrame]:
        """
        Ingest the dataset and yield chunks as pandas DataFrames.
        This enables streaming of arbitrarily large datasets.
        """
