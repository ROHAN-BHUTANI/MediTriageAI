from meditriage.builder.stages.normalize import map_specialty, score_severity
from meditriage.builder.stages.split import assign_split


def test_split_deterministic():
    splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    # Should always return same for same input
    assert assign_split("test_seed_1", splits) == assign_split("test_seed_1", splits)


def test_normalize():
    assert map_specialty("Neurology")[0] == "NEURO"
    assert map_specialty("Unknown")[0] == "GEN_MED"

    assert score_severity("patient had cardiac arrest")[0] == "S1"
    assert score_severity("patient has a fever")[0] == "S3"
