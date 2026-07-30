import io
from unittest.mock import patch

from models.emergent_path_triage.model import EmergentPathTriageModel
from src.data_pipeline import TokenizerPipeline


def test_sequence_length_token_indexing_warning_regression(capsys, caplog):
    """
    Regression test reproducing the sequence-length token indexing failure/warning:
    'Token indices sequence length is longer than the specified maximum sequence length for this model (674 > 512).'
    Verifies that verbose=False suppresses the warning during token length calculations,
    and that TokenizerPipeline properly truncates inputs for inference without indexing errors or crashes.
    """
    tokenizer = EmergentPathTriageModel.build_tokenizer()

    # Create an input sequence longer than 512 tokens (mimicking the 674-token failure case)
    long_text = (
        "patient complaints of severe thoracic pain accompanied by high grade fever and nausea "
        * 80
    )

    # Verify fixed behavior with verbose=False (as fixed in prediction_error_analysis.py and data_pipeline.py)
    with patch("sys.stderr", new=io.StringIO()) as fake_stderr:
        tokens_clean = tokenizer.encode(long_text, verbose=False)
        assert (
            len(tokens_clean) > 512
        ), f"Expected token sequence > 512, got {len(tokens_clean)}"
        stderr_output = fake_stderr.getvalue()
        assert "Token indices sequence length is longer" not in stderr_output
        assert "Token indices sequence length is longer" not in capsys.readouterr().err
        assert "Token indices sequence length is longer" not in capsys.readouterr().out

    # Verify stability during pipeline tokenization (truncation applied to prevent model IndexError)
    pipeline = TokenizerPipeline(tokenizer, max_length=64)
    encoded = pipeline([long_text])
    assert encoded["input_ids"].shape[1] == 64
    assert encoded["attention_mask"].shape[1] == 64
    assert (
        encoded["input_ids"].shape[1] <= 512
    ), "Sequence length must be <= 512 to prevent positional indexing errors in model inference."
