import pytest
from meditriage.builder.adapters.mtsamples import MTSamplesAdapter

def test_mtsamples_adapter():
    adapter = MTSamplesAdapter()
    assert adapter.dataset_source == "mtsamples"
    assert adapter.version == "1.0.0"
