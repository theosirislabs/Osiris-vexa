"""config_test.py — on-demand credential tests behind Settings → Models "Test" buttons.

The two silent-failure modes this surface exists to catch (both bit the owner on 2026-07-09):
  * subscription mode: the mounted ``~/.claude/.credentials.json`` is a STALE export (macOS
    Keychain holds the live token; the file expires ~8-12h) → every turn dies with
    "401 Invalid authentication credentials" and nothing in the UI says why.
  * transcription: a Settings-level override silently outranks ``.env`` (user > global > env)
    and a zero-balance external token 402s every segment while the bot logs stay in docker.

Tests are HONEST about depth: a custom endpoint gets a real 1-token completion; the
subscription file gets an existence + expiry check (the CLI lives only in worker images, so a
live inference test would need a dispatch — the expiry check catches 100% of the observed
failures); the transcription backend gets a real authenticated ``/balance`` probe.

Pure functions over injected fetchers — the routes in api.py are thin wrappers.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

# The compose-mounted subscription credential file (same :ro mount the runtime probes; the
# docker backend mounts the same host path into workers at /root/.claude/.credentials.json).
CREDS_PATH = "/var/lib/vexa/host-claude-credentials"

# The macOS remedy, verbatim — the error message must carry the fix (fail loud AND helpful).
KEYCHAIN_REFRESH = ('security find-generic-password -s "Claude Code-credentials" -w '
                    "> ~/.claude/.credentials.json")

_TIMEOUT = 8.0
# The audio round-trip probe transcribes a real ~1s clip — give the model time to answer.
_STT_PROBE_TIMEOUT = 20.0

# Cloudflare (and similar edges) bot-fight bare urllib/Python clients with HTTP 403 / error 1010
# when no browser-like User-Agent is present — observed against OpenCode Go (opencode.ai/zen/*).
# A stable product UA keeps Save & test honest without looking like a headless scraper.
_PROBE_UA = "VexaConfigTest/0.12 (+https://docs.vexa.ai; model-credential-probe)"

# (status, body_text) — injectable for tests; None body on network failure.
HttpPost = Callable[[str, dict, dict], tuple[int, str]]
HttpGet = Callable[[str, dict], tuple[int, str]]
# (endpoint, token) → (status, body_text) for the STT audio round-trip — injectable for tests.
TranscribeProbe = Callable[[str, str], tuple[int, str]]


def _post(url: str, payload: dict, headers: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": _PROBE_UA, **headers},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def _get(url: str, headers: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        url, headers={"User-Agent": _PROBE_UA, **headers}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def _result(ok: bool, summary: str, **extra) -> dict:
    return {"ok": ok, "summary": summary, **extra}


# ── models ────────────────────────────────────────────────────────────────────────────────────

def test_subscription_credentials(creds_path: str = CREDS_PATH, *, now: Optional[float] = None) -> dict:
    """The mounted credentials file: present → parseable → unexpired. Expiry IS the recurring
    local failure (stale Keychain export), so the failure message ships the exact remedy."""
    if not os.path.isfile(creds_path):
        # docker turns a MISSING host path into an empty dir — same failure, same message.
        return _result(False, "No subscription credentials mounted "
                              "(HOST_CLAUDE_CREDENTIALS unset, or the host file is missing) — "
                              "all setup options: https://docs.vexa.ai/configuration")
    try:
        with open(creds_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return _result(False, "Credentials file is unreadable or not JSON — re-export it: "
                              + KEYCHAIN_REFRESH)
    oauth = data.get("claudeAiOauth") if isinstance(data, dict) else None
    expires_ms = (oauth or data or {}).get("expiresAt") if isinstance(oauth or data, dict) else None
    if not isinstance(expires_ms, (int, float)):
        return _result(False, "Credentials file has no expiresAt — not a Claude Code "
                              "credential export? Re-export it: " + KEYCHAIN_REFRESH)
    left_h = (expires_ms / 1000.0 - (now if now is not None else time.time())) / 3600.0
    if left_h <= 0:
        return _result(False, "Subscription token EXPIRED (stale Keychain export — the known "
                              "macOS gotcha). Refresh with: " + KEYCHAIN_REFRESH,
                       expired=True)
    return _result(True, f"Subscription credentials valid — token expires in {left_h:.1f} h. "
                         "(File check; inference itself runs in workers.)",
                   expires_in_hours=round(left_h, 1))


def test_custom_endpoint(base_url: str, api_key: str, model: str = "",
                         post: HttpPost = _post) -> dict:
    """A REAL 1-token completion against the configured endpoint.

    Tries the two dialects the dispatch overlay brokers (ANTHROPIC_* vs VEXA_LLM_*):

    * Anthropic Messages: ``{origin}/v1/messages``
    * OpenAI-compat: ``{base}/chat/completions`` when ``base`` already ends with ``/v1``
      (OpenRouter, OpenCode Go, OpenAI — what Settings + ``VEXA_LLM_BASE_URL`` store), else
      ``{base}/v1/chat/completions`` for a host-only origin.

    Earlier builds always appended ``/v1/…`` to whatever the operator typed, so a correct
    OpenAI-compat base like ``https://opencode.ai/zen/go/v1`` was probed at
    ``…/v1/v1/chat/completions`` (404) while Cloudflare 403 on the Anthropic path was
    misread as a bad API key. We try every candidate URL and grade the best response.
    """
    base = base_url.rstrip("/")
    if not base:
        return _result(False, "Custom mode but no Base URL set.")
    model = model or "claude-haiku-4-5-20251001"
    auth = {"x-api-key": api_key, "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01"}
    # Origin without a trailing /v1 segment (…/zen/go/v1 → …/zen/go).
    origin = base[:-3] if base.endswith("/v1") else base
    openai_url = (f"{base}/chat/completions" if base.endswith("/v1")
                  else f"{base}/v1/chat/completions")
    responses_url = (f"{base}/responses" if base.endswith("/v1")
                     else f"{base}/v1/responses")
    # A reasoning model spends the output budget before answering, so a 1-token cap returns
    # 200 with NO text — grade on the status, not the body, and give it room to think.
    responses_candidate = ("openai-responses", responses_url,
                           {"model": model, "input": "ping", "max_output_tokens": 1024,
                            "store": False})
    # A deployment on the openai-responses provider is probed there FIRST: the reasoning
    # models it serves (muse-spark-*) reject /chat/completions AND /v1/messages with
    # ModelProtocolUnsupported, so probing those first would misreport a working credential
    # as a broken one. Matches llm/responses_api.py. Offered ONLY for that provider — adding
    # it as a general fallback would outrank the anthropic dialect and change what every
    # other deployment grades.
    on_responses = (os.environ.get("VEXA_LLM_PROVIDER") or "").strip() == "openai-responses"
    candidates = [
        *([responses_candidate] if on_responses else []),
        # OpenAI-compat first when the operator already pasted a /v1 base (the common case
        # for OpenCode Go / OpenRouter / OpenAI) — avoids the double-/v1 trap and matches
        # llm/openai_compat.py which posts to ``{base}/chat/completions``.
        ("openai-compat", openai_url,
         {"model": model, "max_tokens": 1,
          "messages": [{"role": "user", "content": "ping"}]}),
        ("anthropic", f"{origin}/v1/messages",
         {"model": model, "max_tokens": 1,
          "messages": [{"role": "user", "content": "ping"}]}),
    ]
    last_status, last_body, last_label = 0, "", ""
    try:
        for label, url, payload in candidates:
            status, body = post(url, payload, auth)
            last_status, last_body, last_label = status, body, label
            if 200 <= status < 300:
                return _result(True,
                               f"Live completion OK against {base} "
                               f"(model {model}, dialect {label}).",
                               status=status, dialect=label)
            # Auth failures are definitive for that dialect; still try the other shape
            # (OpenCode Go's /messages may 4xx while /chat/completions is fine).
            if status in (401, 403) and label == "openai-compat":
                # Keep going — Anthropic path might clarify; if both auth-fail, report below.
                continue
    except Exception as exc:  # DNS, refused, TLS, timeout — the endpoint itself is the problem
        return _result(False, f"Endpoint unreachable: {exc}")
    detail = last_body[:200] if last_body else ""
    # OpenCode Go (and some gateways) return HTTP 401 for *unsupported model ids* with a
    # ModelError body — that is NOT a bad API key. Surface the body so the operator fixes
    # the model field (e.g. MiMo-V2.5 → mimo-v2.5) instead of rotating a working key.
    lower = (last_body or "").lower()
    if last_status in (401, 403) and (
            "model" in lower and ("not supported" in lower or "modelerror" in lower
                                  or "does not exist" in lower or "unknown model" in lower)):
        return _result(False,
                       f"Model rejected by {base} (HTTP {last_status}): use the exact API "
                       f"model id (e.g. kimi-k2.7-code, mimo-v2.5), not a display name. "
                       f"{detail}",
                       status=last_status)
    if last_status in (401, 403):
        return _result(False, f"Authentication FAILED at {base} (HTTP {last_status}) — bad or "
                              "expired API key, or the edge blocked the probe."
                              + (f" Detail: {detail}" if detail else ""),
                       status=last_status)
    return _result(False, f"Endpoint answered HTTP {last_status} ({last_label}): {detail}",
                   status=last_status)


def run_models_test(config: dict, env: Optional[dict] = None,
                    creds_path: str = CREDS_PATH, post: HttpPost = _post) -> dict:
    """The EFFECTIVE model credential test — same resolution the dispatch overlay applies
    (Settings user > global config already collapsed by admin-api; env is the floor)."""
    env = env if env is not None else dict(os.environ)
    mode = (config.get("mode") or "").strip()
    base_url = (config.get("base_url") or "").strip() or env.get("ANTHROPIC_BASE_URL", "")
    api_key = (config.get("api_key") or "").strip() or env.get("ANTHROPIC_AUTH_TOKEN", "") \
        or env.get("ANTHROPIC_API_KEY", "")
    if mode == "custom" or (not mode and base_url and api_key):
        out = test_custom_endpoint(base_url, api_key, (config.get("model") or "").strip(),
                                   post=post)
        out["mode"] = "custom"
    else:
        out = test_subscription_credentials(creds_path)
        out["mode"] = "subscription"
    # Non-secret provenance so the UI can say WHAT was tested.
    out["config"] = {k: v for k, v in config.items() if k in ("mode", "model", "meeting_model",
                                                              "base_url") and v}
    return out


# ── transcription ─────────────────────────────────────────────────────────────────────────────

# The OpenAI-compatible transcriptions path every consumer agrees on. Appended only when the
# configured URL does not already carry it — the one rule shared with the config.v1 probe
# (deploy/contracts/config.v1/preflight.py:probe_url), the bot's client, and the dictation route.
_STT_PATH = "/v1/audio/transcriptions"

def _transcribe_probe(endpoint: str, token: str) -> tuple:
    """POST the shared audio probe body — the same request the boot preflight makes."""
    from control_plane.config_preflight import audio_probe_body

    content_type, body = audio_probe_body()
    req = urllib.request.Request(
        endpoint, data=body, method="POST",
        headers={"Content-Type": content_type, "Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=_STT_PROBE_TIMEOUT) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def _verify_transcribes(base: str, token: str, source: str, probe: TranscribeProbe,
                        account: str = "") -> dict:
    """Grade the backend by the ONE question the operator is actually asking: will a bot get a
    transcript out of this? Answered by sending real audio — the same request a bot's first chunk
    makes, and the same body the boot preflight sends.

    An EMPTY body cannot answer it: a metered backend answers an empty POST the same way whether
    the credential is funded or worthless, which is how a token that 402s every segment of every
    meeting used to test green here. Nor can the account's balance answer it — a billing-exempt
    account reports 0.0 minutes and transcribes perfectly, so a balance threshold condemns the
    working credential and clears nothing. Sending audio makes the verdict independent of whose
    token it is, so no account identity is named anywhere in this codebase."""
    endpoint = base if base.endswith(_STT_PATH) else base + _STT_PATH
    who = f" ({account})" if account else ""
    try:
        status, body = probe(endpoint, token)
    except Exception as exc:
        return _result(False, f"Backend unreachable: {exc}", source=source)
    if status in (401, 403):
        return _result(False, f"Token REJECTED by {endpoint} (HTTP {status}).", source=source,
                       status=status)
    if status == 402:
        detail = (body or "").strip()[:160]
        return _result(False, f"Token is valid{who} but cannot pay for transcription (HTTP 402) — "
                              f"every segment will fail and meetings will complete with NO "
                              f"transcript. Top up the account or use a token that can transcribe."
                              + (f" Backend said: {detail}" if detail else ""),
                       source=source, status=status, account=account or None)
    if status == 404:
        return _result(False, f"No transcriptions endpoint at {endpoint} (HTTP 404) — check the "
                              "URL shape; some gateways also answer 404 for a rejected key.",
                       source=source, status=status)
    if status >= 500:
        return _result(False, f"Backend error at {endpoint} (HTTP {status}).", source=source,
                       status=status)
    return _result(True, f"OK — {endpoint} transcribed the probe clip{who} (HTTP {status}).",
                   source=source, status=status, account=account or None)


def run_transcription_test(url: str, token: str, source: str, get: HttpGet = _get,
                           probe: TranscribeProbe = _transcribe_probe) -> dict:
    """A real round-trip test of the effective STT backend: transcribe a ~1s probe clip with the
    configured token — the same request (and the same probe body) a bot's first chunk and the boot
    preflight make, so the wizard can never green what the deployment refuses.

    The endpoint's own answer is the verdict. Neither of the two indirect oracles survives contact
    with reality: an empty-body POST is answered identically for a funded and a worthless
    credential, and ``balance_minutes`` reads 0.0 for a billing-exempt account that transcribes
    perfectly — the 2026-07-19 recurrence was exactly a zero-balance token probing green while a
    zero-balance *exempt* token did the actual work. Sending audio asks the real question and keeps
    every account identity out of this codebase. ``/balance`` is still consulted first, but only to
    NAME the account in the verdict (a courtesy, never the oracle)."""
    base = (url or "").strip().rstrip("/")
    if not base:
        return _result(False, "No transcription backend configured at any level "
                              "(user, global, or deployment env).", source=source)
    # Bots post to {url}/v1/audio/transcriptions — strip the path for the account lookup.
    probe_base = base
    for suffix in ("/v1/audio/transcriptions", "/v1/audio", "/v1"):
        if probe_base.endswith(suffix):
            probe_base = probe_base[: -len(suffix)]
            break
    if not token:
        return _result(False, f"Backend {probe_base} configured but NO token set ({source}).",
                       source=source)
    account = ""
    try:
        status, body = get(f"{probe_base}/balance", {"X-API-Key": token})
        if status == 200:
            try:
                account = (json.loads(body) or {}).get("email") or ""
            except ValueError:
                account = ""
    except Exception:
        pass  # no /balance ⇒ not a Vexa gateway; the round-trip below is the oracle either way
    return _verify_transcribes(base, token, source, probe, account)
