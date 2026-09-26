"""responses_api.py — the OpenAI **Responses** dialect (``POST {base}/responses``) as a third
CompletionPort adapter.

Why a third dialect when ``openai_compat`` already speaks ``/chat/completions``: the two are NOT
interchangeable on every gateway. Reasoning models served by OpenCode Zen (``muse-spark-*-contributor``)
reject ``/chat/completions`` with ``ModelProtocolUnsupported`` and reject the Anthropic Messages
dialect the same way — they are reachable ONLY over ``/responses``. Pointing ``openai-compat`` at such
a model therefore cannot work, at any key or privacy setting; the endpoint itself is the difference.
This adapter is that endpoint, and nothing else changes for a deployment already on ``openai-compat``.

Config (constructor args win over env): ``VEXA_LLM_BASE_URL`` (required — e.g.
``https://opencode.ai/zen/go/v1``), ``VEXA_LLM_API_KEY`` (falls back ``ANTHROPIC_AUTH_TOKEN`` →
``ANTHROPIC_API_KEY``; optional for local runtimes), ``VEXA_LLM_MODEL`` (the deployment default),
``VEXA_LLM_MAX_TOKENS`` (the cross-adapter output cap, default 4096 — reasoning models spend most
of the budget thinking, so a chat-sized cap returns an EMPTY completion), and
``VEXA_LLM_REASONING_EFFORT`` (``minimal|low|medium|high``, default ``medium``; omitted entirely
when unset/``none`` so non-reasoning models are unaffected).

``store`` is sent ``false``: this port carries meeting content, and the Responses default is to
retain the response on the provider. A completion has no use for a server-side copy.
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

import httpx

from llm.errors import LLMAuthError, LLMConfigError, LLMError
from llm.ports import CompletionResult

# Reasoning models bill the reasoning pass against the SAME output budget as the answer, so a
# chat-completion-sized cap yields status=incomplete with zero output text. Generous by default.
_DEFAULT_MAX_OUTPUT_TOKENS = 4096
_VALID_EFFORTS = ("minimal", "low", "medium", "high")
_DEFAULT_EFFORT = "medium"


def _max_output_tokens() -> int:
    raw = (os.environ.get("VEXA_LLM_MAX_TOKENS") or "").strip()
    if not raw:
        return _DEFAULT_MAX_OUTPUT_TOKENS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_OUTPUT_TOKENS
    # The provider rejects anything < 16 with a 400; fail loud here instead of at the edge.
    return value if value >= 16 else _DEFAULT_MAX_OUTPUT_TOKENS


def _reasoning_effort() -> Optional[str]:
    raw = (os.environ.get("VEXA_LLM_REASONING_EFFORT") or "").strip().lower()
    if raw in ("", "none", "off"):
        return None
    return raw if raw in _VALID_EFFORTS else _DEFAULT_EFFORT


def _output_text(payload: dict) -> str:
    """Concatenate ``output_text`` content items. Reasoning items are SKIPPED — a reasoning model
    returns its scratchpad as a separate content type, and shipping that to a card beat would
    surface the model's private reasoning as the user-visible insight."""
    chunks: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "".join(chunks)


class OpenAIResponsesCompletion:
    name = "openai-responses"

    def __init__(self, *, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None, timeout: float = 120.0,
                 transport: Optional[httpx.BaseTransport] = None) -> None:
        self._base = (base_url or os.environ.get("VEXA_LLM_BASE_URL")
                      or os.environ.get("ANTHROPIC_BASE_URL") or "").rstrip("/")
        self._key = (api_key or os.environ.get("VEXA_LLM_API_KEY")
                     or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                     or os.environ.get("ANTHROPIC_API_KEY") or "")
        self._model = model or os.environ.get("VEXA_LLM_MODEL") or ""
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 model: Optional[str] = None) -> CompletionResult:
        target = (model or "").strip() or self._model
        if not self._base:
            raise LLMConfigError(
                "no completion endpoint: set VEXA_LLM_BASE_URL (e.g. https://opencode.ai/zen/go/v1) — "
                "the openai-responses provider has no default host"
            )
        if not target:
            raise LLMConfigError(
                "no model: set VEXA_LLM_MODEL (deployment default) or a model in the workspace's "
                "agents/meeting.md"
            )
        body: dict = {
            "model": target,
            "input": prompt,
            "max_output_tokens": _max_output_tokens(),
            "store": False,
        }
        if system:
            body["instructions"] = system
        effort = _reasoning_effort()
        if effort:
            body["reasoning"] = {"effort": effort}
        # User-Agent: some edges (e.g. OpenCode Go on Cloudflare) 403 bare clients (error 1010).
        # x-opencode-session: OpenCode Go refuses to route a request that omits it.
        headers = {"User-Agent": "vexa-terminal/0.12 (OSIRIS Meet; +https://docs.vexa.ai)",
                   "x-opencode-session": uuid.uuid4().hex}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        try:
            r = self._client.post(f"{self._base}/responses", json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise LLMError(f"completion transport failure against {self._base}: {exc}") from exc
        if r.status_code in (401, 403):
            raise LLMAuthError(f"{r.status_code} from {self._base}: {r.text[:300]}")
        if r.status_code >= 400:
            raise LLMError(f"{r.status_code} from {self._base}: {r.text[:300]}")
        try:
            payload = r.json()
        except ValueError as exc:
            raise LLMError(f"malformed completion payload from {self._base}: {exc}") from exc

        # A 200 can still carry a failed call: this dialect reports provider-side faults in-band
        # ({"error": …}) and burns the budget without output on an incomplete turn. Neither may
        # reach a card beat as empty text — say why instead.
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            raise LLMError(f"{error.get('type') or 'provider'}: {str(error['message'])[:300]}")
        text = _output_text(payload)
        if not text.strip():
            reason = ((payload.get("incomplete_details") or {}).get("reason")
                      or payload.get("status") or "no output")
            usage = payload.get("usage") or {}
            raise LLMError(
                f"empty completion from {self._base} ({reason}) — a reasoning model spends the output "
                f"budget before answering; raise VEXA_LLM_MAX_TOKENS above "
                f"{usage.get('output_tokens') or _max_output_tokens()}"
            )
        return CompletionResult(text=text, model=target)
