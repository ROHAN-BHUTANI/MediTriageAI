"""Unit tests for production LLM providers (GeminiProvider, OpenAIProvider, Resilience).

All API calls are mocked; no live API keys or external network connections required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reconstruction.llm import list_providers
from reconstruction.llm.gemini_provider import GeminiProvider
from reconstruction.llm.openai_provider import OpenAIProvider
from reconstruction.llm.resilience import (
    BatchCheckpointer,
    RateLimitError,
    retry_with_backoff,
    validate_generation,
)

# ─── Resilience & Utility Tests ─────────────────────────────────────────────


class TestResilienceUtilities:
    def test_validate_generation_valid(self):
        assert (
            validate_generation("Patient complains of severe chest pain", "CARDIO_PULM")
            is True
        )

    def test_validate_generation_too_short(self):
        assert validate_generation("short", "CARDIO_PULM") is False

    def test_validate_generation_refusal(self):
        assert (
            validate_generation(
                "I am sorry, as an AI I cannot help with that.", "CARDIO_PULM"
            )
            is False
        )

    def test_validate_generation_empty(self):
        assert validate_generation("", "CARDIO_PULM") is False
        assert validate_generation(None, "CARDIO_PULM") is False

    def test_retry_with_backoff_success(self):
        calls = 0

        def _fn():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ValueError("Transient error")
            return "success"

        res = retry_with_backoff(_fn, max_retries=5, initial_delay=0.01)
        assert res == "success"
        assert calls == 3

    def test_retry_with_backoff_rate_limit(self):
        calls = 0

        def _fn():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RateLimitError(retry_after=0.01)
            return "ok"

        res = retry_with_backoff(_fn, max_retries=3, initial_delay=0.01)
        assert res == "ok"
        assert calls == 2

    def test_retry_with_backoff_exhaustion(self):
        def _fail():
            raise RuntimeError("Persistent error")

        with pytest.raises(RuntimeError, match="Persistent error"):
            retry_with_backoff(_fail, max_retries=2, initial_delay=0.01)

    def test_batch_checkpointer(self, tmp_path: Path):
        ckpt_file = tmp_path / "checkpoint.jsonl"
        ckpt = BatchCheckpointer(ckpt_file)
        assert ckpt.count == 0

        batch1 = [{"id": "s1", "text": "sample 1"}, {"id": "s2", "text": "sample 2"}]
        ckpt.append_batch(batch1)
        assert ckpt.count == 2

        # Test resume / reloading
        ckpt2 = BatchCheckpointer(ckpt_file)
        assert ckpt2.count == 2
        assert ckpt2.samples == batch1

        ckpt2.clear()
        assert not ckpt_file.exists()
        assert ckpt2.count == 0


# ─── Gemini Provider Tests ──────────────────────────────────────────────────


class TestGeminiProvider:
    def test_provider_registration(self):
        assert "gemini" in list_providers()

    def test_missing_api_key_raises(self):
        mock_genai = MagicMock()
        with patch.dict(sys.modules, {"google.generativeai": mock_genai}):
            with patch.dict("os.environ", {}, clear=True):
                provider = GeminiProvider(api_key="")
                with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                    provider._ensure_client()

    def test_gemini_generate_success(self):
        mock_genai = MagicMock()
        mock_model_inst = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Patient has severe knee pain and joint swelling."
        mock_response.candidates = [MagicMock(finish_reason="STOP")]
        mock_response.usage_metadata.total_token_count = 42

        mock_model_inst.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model_inst

        with patch.dict(sys.modules, {"google.generativeai": mock_genai}):
            provider = GeminiProvider(api_key="fake-key", model="gemini-2.0-flash")
            results = provider.generate("Generate complaint", n=2)

            assert len(results) == 2
            assert results[0] == "Patient has severe knee pain and joint swelling."
            assert provider.name == "GeminiProvider"

            meta = provider.provider_metadata()
            assert meta["name"] == "GeminiProvider"
            assert meta["model"] == "gemini-2.0-flash"
            assert meta["total_tokens"] == 84
            assert meta["total_calls"] == 2

    def test_gemini_retry_on_rate_limit(self):
        mock_genai = MagicMock()
        mock_model_inst = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Valid complaint"
        mock_response.candidates = [MagicMock(finish_reason="STOP")]
        mock_response.usage_metadata.total_token_count = 10

        mock_model_inst.generate_content.side_effect = [
            Exception("429 Resource exhausted: rate limit exceeded"),
            mock_response,
        ]
        mock_genai.GenerativeModel.return_value = mock_model_inst

        with patch.dict(sys.modules, {"google.generativeai": mock_genai}):
            provider = GeminiProvider(api_key="fake-key", initial_delay=0.01)
            results = provider.generate("Prompt", n=1)

            assert len(results) == 1
            assert results[0] == "Valid complaint"


# ─── OpenAI Provider Tests ──────────────────────────────────────────────────


class TestOpenAIProvider:
    def test_provider_registration(self):
        assert "openai" in list_providers()

    def test_missing_api_key_raises(self):
        mock_openai_module = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai_module}):
            with patch.dict("os.environ", {}, clear=True):
                provider = OpenAIProvider(api_key="")
                with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                    provider._ensure_client()

    def test_openai_generate_success(self):
        mock_openai_module = MagicMock()
        mock_client = MagicMock()

        mock_choice1 = MagicMock()
        mock_choice1.message.content = "Patient complaint 1"
        mock_choice1.finish_reason = "stop"

        mock_choice2 = MagicMock()
        mock_choice2.message.content = "Patient complaint 2"
        mock_choice2.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice1, mock_choice2]
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 30

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_module.OpenAI.return_value = mock_client

        with patch.dict(sys.modules, {"openai": mock_openai_module}):
            provider = OpenAIProvider(
                api_key="fake-key", model="gpt-4o-mini", batch_size=5
            )
            results = provider.generate("Generate complaints", n=2)

            assert len(results) == 2
            assert results[0] == "Patient complaint 1"
            assert results[1] == "Patient complaint 2"

            meta = provider.provider_metadata()
            assert meta["name"] == "OpenAIProvider"
            assert meta["model"] == "gpt-4o-mini"
            assert meta["total_tokens"] == 50
            assert meta["total_calls"] == 1

    def test_openai_batching(self):
        mock_openai_module = MagicMock()
        mock_client = MagicMock()

        def side_effect_completions(model, messages, temperature, max_tokens, n):
            mock_res = MagicMock()
            choices = []
            for i in range(n):
                c = MagicMock()
                c.message.content = f"Sample complaint {i}"
                c.finish_reason = "stop"
                choices.append(c)
            mock_res.choices = choices
            mock_res.usage.prompt_tokens = 10
            mock_res.usage.completion_tokens = 10
            return mock_res

        mock_client.chat.completions.create.side_effect = side_effect_completions
        mock_openai_module.OpenAI.return_value = mock_client

        with patch.dict(sys.modules, {"openai": mock_openai_module}):
            # Batch size 2, requested 5 -> 3 API calls (2 + 2 + 1)
            provider = OpenAIProvider(api_key="fake-key", batch_size=2)
            results = provider.generate("Prompt", n=5)

            assert len(results) == 5
            assert mock_client.chat.completions.create.call_count == 3
