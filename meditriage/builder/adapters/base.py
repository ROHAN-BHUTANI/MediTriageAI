import pandas as pd
from abc import ABC, abstractmethod

class BaseAdapter(ABC):
    @property
    @abstractmethod
    def dataset_source(self) -> str:
        pass
        
    @property
    @abstractmethod
    def version(self) -> str:
        pass

    @abstractmethod
    def ingest(self, raw_path: str) -> pd.DataFrame:
        pass
