"""L2: the openai-responses completion adapter against a fake transport — request shape (the
/responses URL, auth header, instructions, store=false, the reasoning budget), response parsing
(``output_text`` only — reasoning scratchpad never leaks), the in-band error/incomplete handling the
chat dialect has no equivalent of, and the error taxonomy. No network."""
import json

import httpx
import pytest

from llm import LLMAuthError, LLMConfigError, LLMError
from llm.responses_api import OpenAIResponsesCompletion


def _adapter(handler, **kw):
    kw.setdefault("base_url", "https://llm.example/v1")
    kw.setdefault("api_key", "sk-test")
    kw.setdefault("model", "some-reasoning-model")
    return OpenAIResponsesCompletion(transport=httpx.MockTransport(handler), **kw)


def _ok(text="polished"):
    return httpx.Response(200, json={
        "status": "completed",
        "output": [{"type": "message",
                    "content": [{"type": "output_text", "text": text}]}],
    })


def test_request_shape_and_parse(monkeypatch):
    monkeypatch.delenv("VEXA_LLM_REASONING_EFFORT", raising=False)
    monkeypatch.setenv("VEXA_LLM_MAX_TOKENS", "2048")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["session"] = request.headers.get("x-opencode-session")
        seen["body"] = json.loads(request.content)
        return _ok()

    result = _adapter(handler).complete("clean these lines", system="you are a copilot")
    assert result.text == "polished"
    assert result.model == "some-reasoning-model"
    assert seen["url"] == "https://llm.example/v1/responses"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["session"]  # OpenCode Go refuses to route without it
    assert seen["body"]["model"] == "some-reasoning-model"
    assert seen["body"]["input"] == "clean these lines"
    assert seen["body"]["instructions"] == "you are a copilot"
    assert seen["body"]["max_output_tokens"] == 2048
    assert seen["body"]["store"] is False
    assert "reasoning" not in seen["body"]  # effort unset ⇒ non-reasoning models unaffected


def test_reasoning_effort_sent_when_configured(monkeypatch):
    monkeypatch.setenv("VEXA_LLM_REASONING_EFFORT", "low")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return _ok()

    _adapter(handler).complete("p")
    assert seen["body"]["reasoning"] == {"effort": "low"}


def test_reasoning_scratchpad_is_never_returned():
    """A reasoning item is a separate content type; shipping it as the insight would leak the
    model's private scratchpad into a user-visible card."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "secret chain"}]},
                {"type": "message", "content": [{"type": "output_text", "text": "the insight"}]},
            ],
        })

    out = _adapter(handler).complete("p").text
    assert out == "the insight"
    assert "secret chain" not in out


def test_per_call_model_overrides_default():
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["model"] == "beat-model"
        return _ok("ok")

    assert _adapter(handler).complete("p", model="beat-model").model == "beat-model"


def test_no_key_means_no_auth_header(monkeypatch):
    for var in ("VEXA_LLM_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return _ok("ok")

    adapter = OpenAIResponsesCompletion(base_url="http://localhost:8080/v1", api_key="",
                                        model="local", transport=httpx.MockTransport(handler))
    assert adapter.complete("p").text == "ok"


def test_401_raises_auth_error():
    handler = lambda request: httpx.Response(401, text="User not found.")  # noqa: E731
    with pytest.raises(LLMAuthError) as exc:
        _adapter(handler).complete("p")
    assert "401" in str(exc.value)


def test_5xx_raises_llm_error():
    handler = lambda request: httpx.Response(503, text="overloaded")  # noqa: E731
    with pytest.raises(LLMError):
        _adapter(handler).complete("p")


def test_in_band_provider_error_raises_llm_error():
    """This dialect reports provider faults in-band on a 200 — it must not reach a caller as text."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"type": "server_error",
                                                   "message": "Model is unavailable."}})

    with pytest.raises(LLMError) as exc:
        _adapter(handler).complete("p")
    assert "Model is unavailable" in str(exc.value)


def test_budget_exhausted_names_the_env_var(monkeypatch):
    """A reasoning model that burns the cap returns 200 + status=incomplete + NO output. An empty
    insight card is worse than a loud failure, and the remedy is the token budget."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
            "usage": {"output_tokens": 4096},
        })

    with pytest.raises(LLMError) as exc:
        _adapter(handler).complete("p")
    assert "VEXA_LLM_MAX_TOKENS" in str(exc.value)


def test_missing_base_url_fails_loud(monkeypatch):
    for var in ("VEXA_LLM_BASE_URL", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    adapter = OpenAIResponsesCompletion(base_url="", model="m")
    with pytest.raises(LLMConfigError) as exc:
        adapter.complete("p")
    assert "VEXA_LLM_BASE_URL" in str(exc.value)


def test_missing_model_fails_loud(monkeypatch):
    monkeypatch.delenv("VEXA_LLM_MODEL", raising=False)
    adapter = OpenAIResponsesCompletion(base_url="https://llm.example/v1", model="")
    with pytest.raises(LLMConfigError) as exc:
        adapter.complete("p")
    assert "VEXA_LLM_MODEL" in str(exc.value)


def test_max_output_tokens_floor_and_bad_values(monkeypatch):
    """The provider 400s below 16; a typo'd budget must fall back, not reach the edge."""
    from llm.responses_api import _DEFAULT_MAX_OUTPUT_TOKENS, _max_output_tokens

    monkeypatch.setenv("VEXA_LLM_MAX_TOKENS", "4")
    assert _max_output_tokens() == _DEFAULT_MAX_OUTPUT_TOKENS
    monkeypatch.setenv("VEXA_LLM_MAX_TOKENS", "not-a-number")
    assert _max_output_tokens() == _DEFAULT_MAX_OUTPUT_TOKENS
    monkeypatch.setenv("VEXA_LLM_MAX_TOKENS", "512")
    assert _max_output_tokens() == 512
