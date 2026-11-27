"""Route table loading, model inventory validation, and intent resolution."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

DEFAULT_ROUTES_PATH = Path(__file__).resolve().parent.parent / "config/routes.json"
USER_ROUTES_PATH = Path.home() / ".config/nexus/routes.json"
MODELS_CACHE_PATH = Path.home() / ".cache/nexus/models.json"
DEFAULT_DECISION_MODEL = "opencode/nemotron-3.5-lightning-free"

_PROBE_BACKENDS = ("ollama", "openai-compat", "colibri")
# Local probes bypass environment proxies: a loopback/LAN probe must never
# be sent through an HTTP proxy picked up from the shell.
_HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_ROUTE_PREFIX_RE = re.compile(r"^\s*@([A-Za-z0-9_.-]+)\s*")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_'+-]*")
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}")


@dataclass
class OrchestratorDecision:
    orchestrator: str  # "opencode" | "council" | "agy"
    route_key: str
    route_cfg: dict
    cleaned_prompt: str
    take_charge: bool
    reason: str
    decision_model: str = DEFAULT_DECISION_MODEL
    tokens_in: int = 0
    tokens_out: int = 0
    elapsed_ms: int = 0


def load_routes(path: Optional[Path] = None) -> dict:
    """Load the route table, preferring an explicit path then the user override."""
    if path is not None:
        source = Path(path)
    elif USER_ROUTES_PATH.is_file():
        source = USER_ROUTES_PATH
    else:
        source = DEFAULT_ROUTES_PATH

    with open(source, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "routes" not in data:
        raise ValueError(f"route table at {source} has no 'routes' key")
    return data


def list_opencode_models(
    *,
    refresh_cache: bool = False,
    cache_path: Optional[Path] = None,
    ttl_s: int = 3600,
) -> set[str]:
    """Return the model ids `opencode models` advertises, cached on disk."""
    target = Path(cache_path) if cache_path is not None else MODELS_CACHE_PATH
    now = time.time()

    if not refresh_cache:
        try:
            cached = json.loads(target.read_text(encoding="utf-8"))
            if now - float(cached.get("ts", 0)) <= ttl_s:
                models = cached.get("models")
                if isinstance(models, list):
                    return {str(model) for model in models}
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    try:
        proc = subprocess.run(
            ["opencode", "models"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return set()

    models = {line.strip() for line in proc.stdout.splitlines() if line.strip() and "/" in line}
    if not models:
        return set()

    payload = {"ts": now, "models": sorted(models)}
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    return models


def _base_url(endpoint: str) -> str:
    """Reduce a full endpoint to `scheme://netloc`; empty string when malformed."""
    parts = urllib.parse.urlsplit(str(endpoint or ""))
    if not parts.scheme or not parts.netloc:
        return ""
    return f"{parts.scheme}://{parts.netloc}"


def _normalise_model(name: str) -> str:
    return name[: -len(":latest")] if name.endswith(":latest") else name


def _model_matches(model: str, names: list[str]) -> bool:
    """Exact match, treating `foo` and `foo:latest` as the same model, plus virtual auto routes."""
    if not model:
        return False
    if model == "auto" or model.startswith("auto/"):
        if model in names or "auto" in names:
            return True
        if model.startswith("auto/") and bool(names):
            return True
    if model.startswith("openrouter/openrouter/") and model[len("openrouter/"):] in names:
        return True
    wanted = _normalise_model(model)
    return any(_normalise_model(name) == wanted for name in names if name)


def resolve_route_api_key(route_cfg: dict) -> str:
    """Resolve an optional Bearer token from ``api_key_env``, ``api_key_file``, or ``api_key``."""
    env_name = str(route_cfg.get("api_key_env") or "").strip()
    if env_name:
        val = os.environ.get(env_name, "").strip()
        if val:
            return val
    key_file = str(route_cfg.get("api_key_file") or "").strip()
    if key_file:
        try:
            val = Path(key_file).expanduser().read_text(encoding="utf-8").strip()
            if val:
                return val
        except OSError:
            pass
    literal = str(route_cfg.get("api_key") or "").strip()
    return literal


def _probe_url(base: str, backend: str) -> str:
    return f"{base}/api/tags" if backend == "ollama" else f"{base}/v1/models"


def _model_names(payload: dict, backend: str) -> list[str]:
    if backend == "ollama":
        entries = payload.get("models") or []
        return [str(entry.get("name") or "") for entry in entries if isinstance(entry, dict)]
    entries = payload.get("data") or []
    return [str(entry.get("id") or "") for entry in entries if isinstance(entry, dict)]


def probe_local_endpoints(
    routes_data: Optional[dict] = None,
    *,
    timeout_s: float = 2.5,
) -> dict[str, dict]:
    """Probe local Ollama/OpenAI-compatible routes for reachability and model presence.

    Probes never raise: transport, timeout and JSON errors are folded into the
    returned `detail` field so a caller can report them as validation errors.
    """
    data = routes_data if routes_data is not None else load_routes()
    results: dict[str, dict] = {}

    for key, cfg in (data.get("routes") or {}).items():
        backend = str(cfg.get("backend") or "")
        if backend not in _PROBE_BACKENDS:
            continue
        endpoint = str(cfg.get("endpoint") or "")
        model = str(cfg.get("model") or "")
        info = {
            "backend": backend,
            "endpoint": endpoint,
            "model": model,
            "reachable": False,
            "model_present": False,
            "detail": "",
        }

        base = _base_url(endpoint)
        if not base:
            info["detail"] = f"invalid endpoint '{endpoint}'"
            results[key] = info
            continue

        headers: dict[str, str] = {}
        api_key = resolve_route_api_key(cfg)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(_probe_url(base, backend), headers=headers)
        try:
            with _HTTP_OPENER.open(req, timeout=timeout_s) as response:
                status = int(getattr(response, "status", 200))
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except (OSError, urllib.error.URLError, ValueError, TimeoutError) as exc:
            info["detail"] = str(exc) or type(exc).__name__
            results[key] = info
            continue

        info["reachable"] = True
        names = _model_names(payload, backend) if isinstance(payload, dict) else []
        if backend in ("openai-compat", "colibri") and not names:
            # A bare OpenAI-compatible server may omit the model list; a 200 is enough.
            info["model_present"] = status == 200
            info["detail"] = "endpoint responded; no model list advertised"
        else:
            info["model_present"] = _model_matches(model, names)
            info["detail"] = f"{len(names)} model(s) advertised"
        results[key] = info

    return results


def validate_routes(
    routes_data: Optional[dict] = None,
    *,
    refresh_cache: bool = False,
    available_models: Optional[set[str]] = None,
    check_local: bool = False,
) -> tuple[bool, list[str]]:
    """Check pinned models, decision/council wiring and optionally local endpoints."""
    data = routes_data if routes_data is not None else load_routes()
    routes = data.get("routes", {})

    if available_models is None:
        available_models = list_opencode_models(refresh_cache=refresh_cache)

    errors: list[str] = []
    has_inventory = bool(available_models)
    if not has_inventory:
        errors.append("could not enumerate opencode models (`opencode models` returned nothing)")

    if has_inventory:
        for key, cfg in routes.items():
            if cfg.get("backend", "opencode") != "opencode":
                continue
            model = cfg.get("model", "")
            if not model:
                errors.append(f"route '{key}' has no model pinned")
            elif model not in available_models:
                errors.append(f"route '{key}' pins missing model '{model}'")
            fallback = cfg.get("fallback")
            if fallback and fallback not in available_models:
                errors.append(f"route '{key}' pins missing fallback '{fallback}'")

    decision_model = data.get("decision_model")
    if decision_model and decision_model not in available_models:
        errors.append(f"decision_model pins missing model '{decision_model}'")

    for seat, target_route in (data.get("council_roles") or {}).items():
        if target_route not in routes:
            errors.append(f"council role '{seat}' references unknown route '{target_route}'")

    if check_local:
        for key, info in probe_local_endpoints(data).items():
            if not info["reachable"]:
                errors.append(f"local route '{key}' ({info['backend']}) unreachable: {info['detail']}")
            elif not info["model_present"]:
                errors.append(
                    f"local route '{key}' missing model '{info['model']}' at {info['endpoint']}"
                )

    return (not errors), errors


def _keyword_score(prompt: str, keywords: list[str]) -> int:
    text = prompt.lower()
    words = set(_WORD_RE.findall(text))
    score = 0
    for keyword in keywords:
        needle = keyword.lower()
        if needle in words:
            score += 2
        elif len(needle) >= 4 and needle in text:
            # Substring credit only for keywords long enough not to fire by
            # accident: a 2-char keyword like "re" otherwise matches "here".
            score += 1
    return score


def is_explicit_only(route_key: str, cfg: dict) -> bool:
    """Return whether a route is available only through an explicit selection."""
    if bool(cfg.get("explicit_only")):
        return True
    tier = str(cfg.get("quota_tier", "")).strip().upper().replace("_", "-")
    return not cfg.get("keywords") and tier in ("LOCAL", "PREMIUM", "ARGON", "ARGON-QUOTA")


_FAST_QA_ACTION_RE = re.compile(
    r"\b(?:edit|create|write|refactor|run|commit|test|fix|bug|delete|install|script|code|implement|audit|debug|build|compile|patch|grep|search|file|repo)\b",
    re.IGNORECASE,
)
_FAST_QA_PATH_RE = re.compile(
    r"(?:^|\s)[.~/]?[\w.-]+/[\w.-]+|\b\w+\.(?:py|js|ts|tsx|json|md|rs|go|c|cpp|h|java|sh|tcss|toml|yaml|yml)\b",
    re.IGNORECASE,
)


def _is_simple_qa_prompt(prompt: str) -> bool:
    clean = (prompt or "").strip()
    if not clean or len(clean) > 280 or "```" in clean:
        return False
    if _FAST_QA_PATH_RE.search(clean) or _FAST_QA_ACTION_RE.search(clean):
        return False
    return True


def resolve_route(
    prompt: str,
    explicit_route: Optional[str] = None,
    routes_data: Optional[dict] = None,
) -> tuple[str, dict, str]:
    """Resolve a prompt to (route_key, route_config, cleaned_prompt).

    Precedence: explicit flag, then a leading `@route` prefix, then keyword
    scoring, then optional `fast_qa_route`, then the configured default. Unknown
    explicit keys raise KeyError so a typo can never silently fall back.
    """
    data = routes_data if routes_data is not None else load_routes()
    routes = data.get("routes", {})
    default_key = data.get("default_route", "code")

    if explicit_route is not None:
        if explicit_route not in routes:
            raise KeyError(f"unknown route '@{explicit_route}' (known: {', '.join(sorted(routes))})")
        return explicit_route, dict(routes[explicit_route]), prompt

    match = _ROUTE_PREFIX_RE.match(prompt)
    if match:
        candidate = match.group(1)
        if candidate not in routes:
            raise KeyError(f"unknown route '@{candidate}' (known: {', '.join(sorted(routes))})")
        return candidate, dict(routes[candidate]), prompt[match.end() :]

    best_key = default_key
    best_score = 0
    precedence = {name: idx for idx, name in enumerate(data.get("precedence") or [])}
    best_rank = len(precedence) + 1
    for key, cfg in routes.items():
        if is_explicit_only(key, cfg):
            continue
        score = _keyword_score(prompt, list(cfg.get("keywords") or []))
        rank = precedence.get(key, len(precedence))
        # Ties break towards the more specific route listed earlier in
        # `precedence`, so `triage`/`vision` win over the generic `code`.
        if score > best_score or (score == best_score and score > 0 and rank < best_rank):
            best_key, best_score, best_rank = key, score, rank

    if best_score == 0:
        fast_key = str(data.get("fast_qa_route") or "").strip()
        if fast_key and fast_key in routes and _is_simple_qa_prompt(prompt):
            best_key = fast_key

    return best_key, dict(routes.get(best_key, {})), prompt


def load_budgets(routes_data: Optional[dict] = None) -> dict:
    """Return delegation budgets, defaulting to conservative values."""
    data = routes_data if routes_data is not None else load_routes()
    budgets = dict(data.get("budgets") or {})
    budgets.setdefault("max_spans_per_trace", 32)
    budgets.setdefault("max_children", 8)
    budgets.setdefault("max_depth", 2)
    return budgets


def budget_allows(
    budgets: dict,
    *,
    spans: int = 0,
    children: int = 0,
    depth: int = 0,
) -> tuple[bool, str]:
    """Check a proposed delegation against the configured budgets."""
    if spans >= int(budgets.get("max_spans_per_trace", 32)):
        return False, f"span budget exhausted ({spans}/{budgets['max_spans_per_trace']})"
    if children > int(budgets.get("max_children", 8)):
        return False, f"child fan-out {children} exceeds max_children {budgets['max_children']}"
    if depth >= int(budgets.get("max_depth", 2)):
        return False, f"delegation depth {depth} reached max_depth {budgets['max_depth']}"
    return True, ""


def _parse_decision_output(out_raw: str) -> tuple[str, int, int]:
    """Extract reply text and token usage from `opencode run --format json` output."""
    tok_in = 0
    tok_out = 0
    text_chunks: list[str] = []
    for line in out_raw.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        part = event.get("part") if isinstance(event.get("part"), dict) else event
        if event.get("type") == "text" or part.get("type") == "text":
            text_chunks.append(str(part.get("text") or event.get("text") or ""))
        tokens = event.get("tokens") or part.get("tokens")
        if isinstance(tokens, dict):
            cache = tokens.get("cache")
            cache_read = int(cache.get("read", 0) or 0) if isinstance(cache, dict) else 0
            tok_in = max(tok_in, int(tokens.get("input", 0) or 0) + cache_read)
            tok_out = max(tok_out, int(tokens.get("output", 0) or 0))
    return "".join(text_chunks).strip(), tok_in, tok_out


def _fallback_decision(
    prompt: str,
    data: dict,
    decision_model: str,
    reason: str = "",
) -> OrchestratorDecision:
    """Deterministic keyword fallback used whenever the LLM path is skipped or fails."""
    route_key, route_cfg, cleaned = resolve_route(prompt, routes_data=data)
    return OrchestratorDecision(
        orchestrator="opencode",
        route_key=route_key,
        route_cfg=route_cfg,
        cleaned_prompt=cleaned,
        take_charge=True,
        reason=reason or f"keyword match -> @{route_key}",
        decision_model=decision_model,
        tokens_in=max(1, len(prompt) // 4),
        tokens_out=12,
        elapsed_ms=2,
    )


def _http_front_desk_call(
    endpoint: str,
    model: str,
    classifier_prompt: str,
    *,
    api_key: str = "",
    timeout_s: float = 5.0,
) -> tuple[str, int, int]:
    """Query an OpenAI-compatible HTTP endpoint for single-line front-desk routing JSON."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": classifier_prompt}],
            "max_tokens": 96,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
    with _HTTP_OPENER.open(req, timeout=max(0.5, float(timeout_s))) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))
    content = ""
    choices = data.get("choices") or []
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        content = str(msg.get("content") or choices[0].get("text") or "").strip()
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    tok_in = int(usage.get("prompt_tokens") or 0)
    tok_out = int(usage.get("completion_tokens") or 0)
    return content, tok_in, tok_out


def front_desk_decide(
    prompt: str,
    *,
    explicit_route: Optional[str] = None,
    routes_data: Optional[dict] = None,
    use_llm: bool = True,
    timeout_s: float = 15.0,
    cancel_event: Optional[threading.Event] = None,
    shadow_mode: Optional[bool] = None,
    shadow_log_path: Optional[Path] = None,
) -> OrchestratorDecision:
    """Front-facing free decision model that selects which orchestrator takes charge."""
    data = routes_data if routes_data is not None else load_routes()
    routes = data.get("routes", {})
    decision_model = data.get("decision_model", DEFAULT_DECISION_MODEL)

    # 1. Explicit flag or @prefix bypasses LLM call immediately
    if explicit_route is not None or _ROUTE_PREFIX_RE.match(prompt):
        r_key, r_cfg, cleaned = resolve_route(prompt, explicit_route=explicit_route, routes_data=data)
        orch = "agy" if r_cfg.get("backend") == "agy" else "opencode"
        return OrchestratorDecision(
            orchestrator=orch,
            route_key=r_key,
            route_cfg=r_cfg,
            cleaned_prompt=cleaned,
            take_charge=True,
            reason=f"explicit @{r_key} override",
            decision_model=decision_model,
            tokens_in=max(1, len(prompt) // 4),
            tokens_out=8,
            elapsed_ms=1,
        )

    # 2. Query the front-facing free decision model (direct HTTP if configured, else OpenCode)
    if cancel_event is not None and cancel_event.is_set():
        return _fallback_decision(prompt, data, decision_model, "cancelled before decision")

    def _run_llm_decision() -> tuple[Optional[dict], int, int, int, str]:
        route_catalog = ", ".join(
            f"{k} ({v.get('model')}: {v.get('description', '')})"
            for k, v in routes.items()
            if not is_explicit_only(k, v)
        )
        classifier_prompt = (
            "You are the Nexus Front-Desk Decision Model. Pick the best specialist orchestrator to take charge of the user's task. "
            "Prefer free cloud routes when they fit the task. Output ONLY a single-line JSON object with keys: "
            '{"orchestrator": "opencode"|"council", "route": "<route_key>", "take_charge": true, "reason": "<under 10 words>"}.\n'
            f"Available routes: {route_catalog}, or set orchestrator='council' (route='free-reason') for multi-perspective architecture/security debates.\n"
            f"User request: {prompt[:500]}"
        )
        dec_route_name = str(data.get("decision_route") or "").strip()
        dec_route_cfg = routes.get(dec_route_name) if dec_route_name else None
        dec_endpoint = str(data.get("decision_endpoint") or "").strip()
        use_http = bool(
            (isinstance(dec_route_cfg, dict) and dec_route_cfg.get("backend") in ("openai-compat", "colibri"))
            or dec_endpoint
        )
        active_model = decision_model
        t0 = time.monotonic()
        raw_reply = ""
        tok_in = tok_out = 0

        if use_http:
            cfg_ep = str((dec_route_cfg or {}).get("endpoint") or dec_endpoint)
            active_model = str((dec_route_cfg or {}).get("model") or decision_model)
            api_key = resolve_route_api_key(dec_route_cfg or {})
            try:
                raw_reply, tok_in, tok_out = _http_front_desk_call(
                    cfg_ep,
                    active_model,
                    classifier_prompt,
                    api_key=api_key,
                    timeout_s=min(float(timeout_s), 5.0),
                )
            except Exception:
                raw_reply = ""
        else:
            try:
                from nexus.backends import _run_subprocess_bounded

                _rc, out_raw, _err, _timed_out = _run_subprocess_bounded(
                    [
                        "opencode",
                        "run",
                        "--auto",
                        "--archived",
                        "--format",
                        "json",
                        "--model",
                        decision_model,
                        "--",
                        classifier_prompt,
                    ],
                    timeout_s=timeout_s,
                    cancel_event=cancel_event,
                )
            except Exception:
                out_raw = ""
            raw_reply, tok_in, tok_out = _parse_decision_output(out_raw)

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        match = _JSON_OBJ_RE.search(raw_reply)
        parsed = None
        if match:
            try:
                cand = json.loads(match.group(0))
                if isinstance(cand, dict):
                    parsed = cand
            except json.JSONDecodeError:
                parsed = None
        return parsed, tok_in, tok_out, elapsed_ms, active_model

    if use_llm:
        is_shadow = bool(data.get("decision_shadow")) if shadow_mode is None else bool(shadow_mode)
        if is_shadow:
            kw_dec = _fallback_decision(prompt, data, decision_model)

            def _record_shadow() -> None:
                try:
                    from nexus.assay import record_shadow_decision

                    parsed, _ti, _to, elapsed_ms, active_model = _run_llm_decision()
                    llm_route = ""
                    if isinstance(parsed, dict):
                        cand_r = str(parsed.get("route") or "").lstrip("@")
                        if cand_r in routes and not is_explicit_only(cand_r, routes[cand_r]):
                            llm_route = cand_r
                    record_shadow_decision(
                        prompt=prompt,
                        keyword_route=kw_dec.route_key,
                        llm_route=llm_route or kw_dec.route_key,
                        agreed=bool(llm_route == kw_dec.route_key),
                        elapsed_ms=elapsed_ms,
                        decision_model=active_model,
                        path=shadow_log_path,
                    )
                except Exception:
                    pass

            if shadow_log_path is not None:
                _record_shadow()
            else:
                threading.Thread(target=_record_shadow, daemon=True).start()
            return kw_dec

        parsed, tok_in, tok_out, elapsed_ms, active_model = _run_llm_decision()
        if cancel_event is not None and cancel_event.is_set():
            return _fallback_decision(prompt, data, decision_model, "cancelled before decision")
        if isinstance(parsed, dict):
            orch = str(parsed.get("orchestrator", "opencode")).lower()
            chosen_route = str(parsed.get("route", "code")).lstrip("@")
            if chosen_route not in routes or is_explicit_only(chosen_route, routes[chosen_route]):
                chosen_route = resolve_route(prompt, routes_data=data)[0]
            reason = str(parsed.get("reason") or f"delegated to @{chosen_route}")[:80]
            return OrchestratorDecision(
                orchestrator="council" if orch == "council" else "opencode",
                route_key=chosen_route,
                route_cfg=dict(routes[chosen_route]),
                cleaned_prompt=prompt,
                take_charge=bool(parsed.get("take_charge", True)),
                reason=reason,
                decision_model=active_model,
                tokens_in=tok_in or max(1, len(prompt) // 4),
                tokens_out=tok_out or 16,
                elapsed_ms=elapsed_ms,
            )

    # 3. Fast deterministic keyword fallback
    return _fallback_decision(prompt, data, decision_model)
