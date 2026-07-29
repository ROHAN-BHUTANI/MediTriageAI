import pytest
from meditriage.builder.adapters.mtsamples import MtsamplesAdapter

def test_mtsamples_adapter():
    adapter = MtsamplesAdapter()
    assert adapter.dataset_source == "mtsamples"
    assert adapter.version == "1.0.0"
