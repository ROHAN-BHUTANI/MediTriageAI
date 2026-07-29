import pandas as pd
class BaseAdapter:
    @property
    def dataset_source(self) -> str:
        raise NotImplementedError
    @property
    def version(self) -> str:
        return "1.0.0"
    def ingest(self, raw_path: str) -> pd.DataFrame:
        raise NotImplementedError
