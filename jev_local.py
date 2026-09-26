"""jev_local v2 - Jev-style typed decisions from a local model's real token probabilities.

Runs against stock llama-server (the official prebuilt release; no fork). Two modes:

  independent  each question is answered in its own context (document + that question).
               Between questions the server rewinds to the end of the document, so the
               document is read once and each question costs its own ~30-60 tokens.
  dependent    one JSON form; each field's answer is conditioned on the earlier fields'
               chosen answers. Only decision tokens are computed; every token before and
               between decisions (keys, quotes, commas, shared prefixes) is injected.

In both modes a decision costs one forward pass: the request injects everything known
up to the decision point as prompt (string pieces + exact token IDs of earlier
decisions) with n_predict=1, and llama-server returns the raw-logit probabilities at
that position. The best legal token is appended and the next step follows until one
answer remains. Nothing after the last decision is computed.

Answer types: boolean (Jev noul), enum of strings/ints/bools (Jev choice; Jev score =
levels 0..n-1), bounded integer (<=200 values, treated as enum), number (each digit is a
decision; returns the value and its path probability). Free strings are refused.

Probabilities: at each step the distribution is restricted to legal tokens (character
match against the remaining answers) and renormalised; alternative tokenisations of the
same answer are summed; alternatives that are still ambiguous (`very` -> "very
negative" / "very positive") are expanded from their exact token-ID path. Verified
against exhaustive brute force.

Hybrid models (Qwen3.8 = Gated-DeltaNet + attention): the recurrent state can only be
rewound to a saved checkpoint. jev sends the prompt as pieces split at the end of the
document and declares that boundary text via `message_delimiters`, so llama-server
keeps a checkpoint exactly there. The model sees the same characters either way.

Usage
  python jev_local.py test
  python jev_local.py serve [--port 8100]     # Jev-compatible POST /v1/systemone

Environment
  LLAMA_URL            llama-server                  default http://127.0.0.1:5005
  LLAMA_API_KEY        only if llama-server has --api-key
  JEV_TOP_LOGPROBS     candidates read per decision  default 50
  JEV_MIN_BRANCH_MASS  skip ambiguous branches below this mass   default 1e-4
  JEV_PREFIX_CHECK     off | first | full: measure how likely the model was to produce the
                       injected opening itself (debug prefix_prob)   default first
  JEV_SERVER_KEY       Bearer key required by `serve`
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    TypeSafeError,
    Usage,
)

__all__ = ["JevLocal", "LocalResponse", "Noul", "Score", "Choice"]

LOCAL_MODEL_NAME = "jev-local"
MAX_INT_ENUM = 200
MAX_EXPANSIONS = 32  # extra requests per field for ambiguous branches
_SPLIT = "\x00JEV_SPLIT\x00"

# All prompt wording comes verbatim from TypeSafe's own LLM adapter (system-one-adapter 0.2.1,
# MIT), discrete-answer mode: the model returns one allowed value per question, which is what
# jev-local reads probabilities from. Imported rather than copied so it can't drift.
# One deviation: the adapter puts the schema instruction in the system prompt, before the
# document; jev-local puts the same text right after the document, so the document can stay
# cached while questions change.
from system_one_adapter._client import _DISCRETE_SYSTEM_PROMPT as SYSTEM_PROMPT  # noqa: E402
from system_one_adapter._client import _OUTPUT_SCHEMA_INSTRUCTION_TEMPLATE as SCHEMA_TEMPLATE  # noqa: E402
from system_one_adapter._client import _serialize_state_as_user_prompt as serialize_document  # noqa: E402
from system_one_adapter import _schema as _adapter_schema  # noqa: E402
from pydantic_core import to_json as _to_json  # noqa: E402  (the adapter's serializer)

# Boundary between the document and the question-specific text, declared to llama-server as a
# message delimiter so a checkpoint is saved exactly there. It is the adapter's own wording.
DELIMITER = "\n\n" + SCHEMA_TEMPLATE.split("{schema}")[0].rstrip("\n")
ANSWERS_DESCRIPTION = ("Exactly one answer per property below. Use these property names verbatim and do not add, "
                       "rename, or nest them under any other key.")  # adapter's TypeSafeAnswers docstring; checked in tests.py
NO_INSTRUCTIONS = "No additional instructions."  # adapter's placeholder for a missing description

MAX_INT_DIGITS, MAX_FRAC_DIGITS = 15, 6
_NUM_PREFIX = re.compile(r"-?(\d{1,%d}(\.\d{0,%d})?)?" % (MAX_INT_DIGITS, MAX_FRAC_DIGITS))
_NUM_FULL = re.compile(r"-?\d{1,%d}(\.\d{1,%d})?" % (MAX_INT_DIGITS, MAX_FRAC_DIGITS))


class LocalResponse(SystemOneResponse):
    """SDK-compatible response. Native (non-Jev) answers are plain dicts."""

    answers: dict[str, Any]  # type: ignore[assignment]
    form: dict[str, Any] = {}
    debug: dict[str, Any] = {}


# --------------------------------------------------------------------------- questions


@dataclass
class _Spec:
    key: str
    kind: str  # boolean | enum | number
    jev: str | None  # noul | choice | score | None
    values: list[Any]
    prop: dict[str, Any]  # this question's JSON-schema property, as shown to the model
    source: Any = None


def _text(v: Any) -> str:
    if v is None:
        return ""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _adapter_property(key: str, q: Any) -> dict[str, Any]:
    """The adapter's own discrete-mode schema property for a Jev question (description text included)."""
    prepared = _adapter_schema.convert_question_collection_to_validated_api_question_models({key: q})
    schema = _adapter_schema.create_raw_output_schema(_adapter_schema.create_llm_output_model(prepared, "discrete"))
    return schema["$defs"]["TypeSafeAnswers"]["properties"][key]


def _spec(key: str, q: Any) -> _Spec:
    if isinstance(q, Mapping):
        t = q.get("type")
        if t in ("noul", "choice", "score"):
            q = {"noul": Noul, "choice": Choice, "score": Score}[t].model_validate(dict(q))
        else:
            return _native_spec(key, q)
    if isinstance(q, Noul):
        return _Spec(key, "boolean", "noul", [True, False], _adapter_property(key, q), q)
    if isinstance(q, Choice):
        return _Spec(key, "enum", "choice", list(q.criteria), _adapter_property(key, q), q)
    if isinstance(q, Score):
        return _Spec(key, "enum", "score", list(range(len(q.criteria))), _adapter_property(key, q), q)
    raise ValueError(f'Question "{key}" must be Noul, Choice, Score, or a typed dict')


def _native_spec(key: str, q: Mapping[str, Any]) -> _Spec:
    """JSON-schema-like questions. The property shown to the model is standard JSON Schema in the
    adapter's style: a description (the caller's text, or the adapter's placeholder) plus type
    keywords. Free strings are refused."""
    desc = _text(q.get("instructions") or q.get("description")) or NO_INSTRUCTIONS
    t = q.get("type")
    if "enum" in q or t == "enum":
        values = list(q.get("enum") or [])
        if not values or any(not isinstance(v, (str, int, float, bool)) for v in values):
            raise ValueError(f'Question "{key}": enum needs a non-empty list of strings/numbers/booleans')
        prop: dict[str, Any] = {"description": desc, "enum": values}
        kinds = {("boolean" if isinstance(v, bool) else "integer" if isinstance(v, int) else "number" if isinstance(v, float) else "string") for v in values}
        if kinds == {"integer", "number"}:
            kinds = {"number"}
        if len(kinds) == 1:
            prop["type"] = kinds.pop()
        return _Spec(key, "enum", None, values, prop, q)
    if t == "boolean":
        return _Spec(key, "boolean", None, [True, False], {"description": desc, "type": "boolean"}, q)
    if t == "integer" and "minimum" in q and "maximum" in q and int(q["maximum"]) - int(q["minimum"]) < MAX_INT_ENUM:
        lo, hi = int(q["minimum"]), int(q["maximum"])
        return _Spec(key, "enum", None, list(range(lo, hi + 1)), {"description": desc, "maximum": hi, "minimum": lo, "type": "integer"}, q)
    if t in ("number", "integer"):
        return _Spec(key, "number", None, [], {"description": desc, "type": t}, q)
    if t == "string":
        raise TypeSafeError(f'Question "{key}": free-text strings are not supported; use an enum.')
    raise ValueError(f'Question "{key}": unsupported type {t!r} (use boolean, enum, integer, number)')


def schema_for(specs: list[_Spec]) -> dict[str, Any]:
    """The adapter's output schema shape: {"answers": {<key>: <value>, ...}} (keys in question order)."""
    return {
        "$defs": {"TypeSafeAnswers": {
            "additionalProperties": False,
            "description": ANSWERS_DESCRIPTION,
            "properties": {s.key: s.prop for s in specs},
            "required": [s.key for s in specs],
            "type": "object",
        }},
        "additionalProperties": False,
        "properties": {"answers": {"$ref": "#/$defs/TypeSafeAnswers"}},
        "required": ["answers"],
        "type": "object",
    }


ANSWERS_OPEN = '{"answers": {'
ANSWERS_CLOSE = "}}"


def independent_fixed(spec: _Spec) -> str:
    """Text that opens an independent answer, e.g. `{"answers": {"is_urgent": `."""
    return ANSWERS_OPEN + json.dumps(spec.key, ensure_ascii=False) + ": "


# --------------------------------------------------------------------------- grammar


def _lit(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _enum_grammar(targets: list[tuple[str, int]], consumed: str) -> str:
    rest = sorted({t[len(consumed):] for t, _ in targets if t.startswith(consumed) and len(t) > len(consumed)})
    return "root ::= " + " | ".join(_lit(r) for r in rest) + "\n"


def _number_grammar(consumed: str, term: str) -> str:
    """Grammar for the remainder of a (bounded) JSON number followed by `term`."""
    t = _lit(term)
    for k in range(1, len(term)):  # part of a multi-character terminator already emitted
        if consumed.endswith(term[:k]) and _NUM_FULL.fullmatch(consumed[:-k]):
            return f"root ::= {_lit(term[k:])}\n"
    m = re.fullmatch(r"(-?)(\d*)(\.(\d*))?", consumed)
    if m is None:
        raise TypeSafeError(f"bad number prefix {consumed!r}")
    sign, ints, has_dot, fracs = m.group(1), m.group(2), m.group(3) is not None, m.group(4) or ""
    frac = lambda lo, hi: f"[0-9]{{{lo},{hi}}}" if hi > 0 else ""  # noqa: E731
    if has_dot:
        need = 1 if not fracs else 0
        body = f"{frac(need, MAX_FRAC_DIGITS - len(fracs))} {t}" if MAX_FRAC_DIGITS - len(fracs) > 0 else t
    else:
        more_int = MAX_INT_DIGITS - len(ints)
        int_part = f"[0-9]{{{1 if not ints else 0},{more_int}}} " if more_int > 0 else ""
        sign_part = '"-"? ' if not sign and not ints else ""
        body = f'{sign_part}{int_part}("." [0-9]{{1,{MAX_FRAC_DIGITS}}})? {t}'
    return f"root ::= {body}\n"


def _number_legal(consumed: str, tok: str, term: str) -> bool:
    """consumed + tok must be a prefix of <number><term> (the terminator may span several tokens)."""
    s = consumed + tok
    if _NUM_PREFIX.fullmatch(s):
        return True
    return any(s.endswith(term[:k]) and _NUM_FULL.fullmatch(s[:-k]) for k in range(1, len(term) + 1))


# --------------------------------------------------------------------------- client


@dataclass
class _Stats:
    requests: int = 0
    prompt_tokens_processed: int = 0
    tokens_decoded: int = 0
    per_request_prompt_n: list[int] = field(default_factory=list)
    legal_mass: float | None = None
    injected: list[str] = field(default_factory=list)
    unexpanded_mass: float = 0.0
    accounted_mass: float | None = None
    decided_after: str | None = None
    prefix_prob: float | None = None
    prefix_checked: str | None = None


class JevLocal:
    """Talks to llama-server directly. Requests are sequential on one slot."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        top_logprobs: int | None = None,
        min_branch_mass: float | None = None,
        prefix_check: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("LLAMA_URL") or "http://127.0.0.1:5005").rstrip("/")
        self.api_key = api_key or os.environ.get("LLAMA_API_KEY") or ""
        self.top_logprobs = top_logprobs or int(os.environ.get("JEV_TOP_LOGPROBS", "50"))
        self.min_branch_mass = float(os.environ.get("JEV_MIN_BRANCH_MASS", "1e-4")) if min_branch_mass is None else min_branch_mass
        self.prefix_check = (prefix_check or os.environ.get("JEV_PREFIX_CHECK", "first")).lower()
        if self.prefix_check not in ("off", "first", "full"):
            raise ValueError("prefix_check must be off, first or full")
        self.timeout = timeout
        self._tok: dict[str, list[int]] = {}
        self._detok: dict[tuple[int, ...], str] = {}

    # ---------------------------------------------------------------- transport
    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + path, json.dumps(body).encode(), headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise TypeSafeError(f"llama-server HTTP {e.code}: {e.read()[:500].decode(errors='replace')}") from e
        except urllib.error.URLError as e:
            raise TypeSafeError(f"Cannot reach llama-server at {self.base_url}: {e.reason}") from e

    def _tokenize(self, text: str) -> list[int]:
        if text not in self._tok:
            self._tok[text] = self._post("/tokenize", {"content": text, "add_special": False})["tokens"]
        return self._tok[text]

    def _detokenize(self, ids: tuple[int, ...]) -> str:
        if ids not in self._detok:
            self._detok[ids] = self._post("/detokenize", {"tokens": list(ids)})["content"] if ids else ""
        return self._detok[ids]

    def _base_pieces(self, state: Any, specs: list[_Spec]) -> list[Any]:
        """[chat-template head ending with the document, DELIMITER as token IDs, schema instruction + assistant header].

        The model reads: system = adapter's discrete system prompt; user = adapter's document block,
        then the adapter's schema instruction with the schema for these questions."""
        instruction = SCHEMA_TEMPLATE.format(schema=_to_json(schema_for(specs)).decode())
        assert ("\n\n" + instruction).startswith(DELIMITER)
        user = serialize_document(state) + _SPLIT + ("\n\n" + instruction)[len(DELIMITER):]
        rendered = self._post("/apply-template", {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        })["prompt"]
        head, tail = rendered.split(_SPLIT, 1)
        # the delimiter goes in as token IDs: same tokenisation the server uses to match
        # `message_delimiters`, and it makes the prompt a mixed array (= one prompt, not several)
        return [head, *self._tokenize(DELIMITER), tail]

    def _complete(self, pieces: list[Any], grammar: str, stats: _Stats) -> dict:
        """One request, n_predict=1: prefill the pieces, return the decision entry."""
        prompt = [p for p in pieces if p != ""]  # strings are tokenised by the server, ints are token IDs
        if all(isinstance(p, str) for p in prompt):  # an all-string array would mean several prompts
            prompt = [prompt[0], *self._tokenize("".join(prompt[1:]))] if len(prompt) > 1 else prompt[0]
        data = self._post("/completion", {
            "prompt": prompt,
            "message_delimiters": [{"role": "user", "delimiter": DELIMITER}],
            "n_predict": 1,
            "grammar": grammar,
            "n_probs": self.top_logprobs,
            "post_sampling_probs": False,
            "top_k": 1,
            "cache_prompt": True,
            "stream": False,
        })
        t = data.get("timings") or {}
        stats.requests += 1
        n = int(t.get("prompt_n") or 0)
        stats.prompt_tokens_processed += n
        stats.per_request_prompt_n.append(n)
        stats.tokens_decoded += int(t.get("predicted_n") or 0)
        entries = data.get("completion_probabilities") or []
        if not entries:
            raise TypeSafeError("llama-server returned no token probabilities.")
        return entries[0]

    @staticmethod
    def _candidates(entry: dict) -> dict[tuple[int, str], float]:
        cands: dict[tuple[int, str], float] = {}
        for c in entry.get("top_logprobs") or []:
            if c.get("logprob") is not None:
                k = (c["id"], c["token"])
                cands[k] = cands.get(k, 0.0) + math.exp(c["logprob"])
        k = (entry["id"], entry.get("token", ""))
        if k not in cands and entry.get("logprob") is not None:
            cands[k] = math.exp(entry["logprob"])
        return cands

    # ---------------------------------------------------------------- enum / boolean decision
    def _plan(self, fixed: str, spec: _Spec, term: str) -> tuple[tuple[int, ...], str, list[tuple[str, int]]]:
        """Tokenise every complete answer text (fixed prefix + value + term); inject the
        longest token prefix they all share; return (inject_ids, inject_text, remaining targets)."""
        full = [fixed + json.dumps(v, ensure_ascii=False) + term for v in spec.values]
        seqs = [self._tokenize(f) for f in full]
        n = 0
        while all(len(s) > n for s in seqs) and len({s[n] for s in seqs}) == 1:
            n += 1
        inject = tuple(seqs[0][:n])
        text = self._detokenize(inject)
        while inject and not all(f.startswith(text) for f in full):  # token straddled a char
            inject = inject[:-1]
            text = self._detokenize(inject)
        return inject, text, [(f[len(text):], i) for i, f in enumerate(full)]

    def _descend(self, base, targets, consumed, ids, alive, path, probs, pending, stats) -> tuple[str, tuple[int, ...]]:
        """Decode one token per request along the best legal path until one value remains.
        Returns (consumed text, decision ids) of the finished path."""
        while True:
            live = [targets[j] for j in alive]
            entry = self._complete(base + list(ids), _enum_grammar(live, consumed), stats)
            legal: dict[tuple[int, str], tuple[float, set[int]]] = {}
            for (tid, tok), p in self._candidates(entry).items():
                if not tok:
                    continue
                hit = {j for j in alive if targets[j][0].startswith(consumed + tok)}
                if hit:
                    legal[(tid, tok)] = (p, hit)
            z = sum(p for p, _ in legal.values())
            if stats.legal_mass is None:
                stats.legal_mass = z
            sampled = (entry["id"], entry.get("token", ""))
            if sampled not in legal or z <= 0:
                raise TypeSafeError(f"sampled token {sampled[1]!r} is not a legal continuation of {consumed!r}")
            for (tid, tok), (p, hit) in legal.items():
                if (tid, tok) == sampled:
                    continue
                values = {targets[j][1] for j in hit}
                m = path * p / z
                if len(values) == 1:
                    probs[values.pop()] += m
                else:
                    key = ids + (tid,)
                    old = pending.get(key)
                    pending[key] = ((old[0] if old else 0.0) + m, hit, consumed + tok)
            p, alive = legal[sampled]
            path *= p / z
            consumed += sampled[1]
            ids = ids + (sampled[0],)
            if len({targets[j][1] for j in alive}) == 1:
                probs[targets[next(iter(alive))][1]] += path
                return consumed, ids

    def _prefix_prob(self, base: list[Any], inject: tuple[int, ...], stats: _Stats) -> None:
        """Probability that the model, unconstrained, would itself write the injected tokens.

        Stock llama-server has no prompt-logprob option, so the injected tokens are generated
        once under a token-level grammar that allows exactly them (<[id]> <[id]> ...). For each
        generated token llama-server reports its raw probability from the unmodified logits,
        exact even when it is far outside the top-k. The product is the probability.

        "first" checks only the first injected token (e.g. `{"`): no extra forward pass,
        because that token's distribution comes out of the prefill the decision needs anyway.
        "full" checks every injected token up to the first decision: len(inject) - 1 extra
        decode steps, all in one request. The generated tokens stay in the prompt cache, so
        the decision request that follows reuses them."""
        if self.prefix_check == "off" or not inject:
            return
        n = len(inject) if self.prefix_check == "full" else 1
        data = self._post("/completion", {
            "prompt": base,
            "message_delimiters": [{"role": "user", "delimiter": DELIMITER}],  # keep the document checkpoint
            "n_predict": n,
            "grammar": "root ::= " + " ".join(f"<[{t}]>" for t in inject[:n]) + "\n",
            "n_probs": 1,
            "post_sampling_probs": False,
            "top_k": 1,
            "cache_prompt": True,
            "stream": False,
        })
        t = data.get("timings") or {}
        stats.requests += 1
        stats.prompt_tokens_processed += int(t.get("prompt_n") or 0)
        stats.per_request_prompt_n.append(int(t.get("prompt_n") or 0))
        stats.tokens_decoded += int(t.get("predicted_n") or 0)
        entries = data.get("completion_probabilities") or []
        if [e["id"] for e in entries] != list(inject[:n]):
            raise TypeSafeError("prefix check: forced generation did not reproduce the injected tokens")
        stats.prefix_prob = math.exp(sum(e["logprob"] for e in entries))
        stats.prefix_checked = self._detokenize(tuple(inject[:n]))

    def _decide_enum(self, base: list[Any], fixed: str, spec: _Spec, term: str, stats: _Stats, check_prefix: bool = False) -> tuple[list[float], int, list[Any], str]:
        """Returns (probabilities, chosen index, prompt pieces through the decision, text still owed)."""
        inject, inject_text, targets = self._plan(fixed, spec, term)
        stats.injected.append(inject_text)
        if check_prefix:
            self._prefix_prob(base, inject, stats)
        probs = [0.0] * len(spec.values)
        pending: dict[tuple[int, ...], tuple[float, set[int], str]] = {}
        base2 = base + list(inject)
        consumed, ids = self._descend(base2, targets, "", (), set(range(len(targets))), 1.0, probs, pending, stats)
        stats.decided_after = inject_text + consumed
        chosen = next(i for t, i in targets if t.startswith(consumed))
        while pending:
            key = max(pending, key=lambda k: pending[k][0])
            mass, hit, prefix = pending.pop(key)
            if mass < self.min_branch_mass or stats.requests > MAX_EXPANSIONS:
                stats.unexpanded_mass += mass
                continue
            self._descend(base2, targets, prefix, key, hit, mass, probs, pending, stats)
        total = sum(probs)
        stats.accounted_mass = total
        dist = [p / total for p in probs] if total > 0 else probs
        owed = targets[chosen][0][len(consumed):]  # rest of the chosen value + term, not yet in the prompt
        return dist, chosen, base2 + list(ids), owed

    # ---------------------------------------------------------------- number decision
    def _decide_number(self, base: list[Any], fixed: str, term: str, stats: _Stats, check_prefix: bool = False) -> tuple[Any, float, list[Any]]:
        stats.injected.append(fixed)
        inject = tuple(self._tokenize(fixed))
        if check_prefix:
            self._prefix_prob(base, inject, stats)
        pieces = base + list(inject)
        consumed, path, ids = "", 1.0, ()
        for _ in range(40):
            entry = self._complete(pieces + list(ids), _number_grammar(consumed, term), stats)
            legal: dict[str, float] = {}
            for (_tid, tok), p in self._candidates(entry).items():
                if tok and _number_legal(consumed, tok, term):
                    legal[tok] = legal.get(tok, 0.0) + p
            z = sum(legal.values())
            if stats.legal_mass is None:
                stats.legal_mass = z
            tok = entry.get("token", "")
            if tok not in legal or z <= 0:
                raise TypeSafeError(f"illegal number token {tok!r} after {consumed!r}")
            path *= legal[tok] / z
            consumed += tok
            ids = ids + (entry["id"],)
            if consumed.endswith(term):
                break
        else:
            raise TypeSafeError("number did not terminate within 40 tokens")
        raw = consumed[: -len(term)]
        value = int(raw) if re.fullmatch(r"-?\d+", raw) else float(raw)
        stats.decided_after = fixed + consumed
        return value, path, pieces + list(ids)

    # ---------------------------------------------------------------- modes
    def _independent(self, state: Any, specs: list[_Spec]) -> tuple[dict[str, Any], dict[str, _Stats]]:
        results: dict[str, Any] = {}
        stats: dict[str, _Stats] = {}
        for spec in specs:
            st = _Stats()
            base = self._base_pieces(state, [spec])
            if spec.kind == "number":
                value, p, _ = self._decide_number(base, independent_fixed(spec), ANSWERS_CLOSE, st, check_prefix=True)
                results[spec.key] = {"kind": "number", "value": value, "probability": p}
            else:
                dist, chosen, _, _ = self._decide_enum(base, independent_fixed(spec), spec, ANSWERS_CLOSE, st, check_prefix=True)
                results[spec.key] = {"kind": spec.kind, "probs": dist, "chosen": chosen}
            stats[spec.key] = st
        return results, stats

    def _dependent(self, state: Any, specs: list[_Spec]) -> tuple[dict[str, Any], dict[str, _Stats]]:
        base = self._base_pieces(state, specs)
        results: dict[str, Any] = {}
        stats: dict[str, _Stats] = {}
        owed = ANSWERS_OPEN  # text known but not yet in the prompt
        for n, spec in enumerate(specs):
            st = _Stats()
            last = n == len(specs) - 1
            term = ANSWERS_CLOSE if last else ","
            fixed = owed + json.dumps(spec.key) + ": "
            check = n == 0 or self.prefix_check == "full"
            if spec.kind == "number":
                value, p, base = self._decide_number(base, fixed, term, st, check_prefix=check)
                results[spec.key] = {"kind": "number", "value": value, "probability": p}
                owed = " "
            else:
                dist, chosen, base, owed = self._decide_enum(base, fixed, spec, term, st, check_prefix=check)
                results[spec.key] = {"kind": spec.kind, "probs": dist, "chosen": chosen}
                owed = owed + " "  # rest of value + terminator, then the space before the next key
            stats[spec.key] = st
        return results, stats

    # ---------------------------------------------------------------- public
    def system_one(self, state: Any, questions: Mapping[str, Any], *, mode: str = "independent", model: str | None = None) -> LocalResponse:
        if state is None:
            raise ValueError("state is required")
        if not questions:
            raise ValueError("at least one question is required")
        if mode not in ("independent", "dependent"):
            raise ValueError("mode must be 'independent' or 'dependent'")
        specs = [_spec(k, q) for k, q in questions.items()]
        started = time.perf_counter()
        results, stats = (self._independent if mode == "independent" else self._dependent)(state, specs)
        answers: dict[str, Any] = {}
        form: dict[str, Any] = {}
        debug: dict[str, Any] = {"mode": mode, "per_question": {}}
        tin = tout = 0
        for spec in specs:
            res, st = results[spec.key], stats[spec.key]
            answers[spec.key] = self._format(spec, res)
            form[spec.key] = res["value"] if spec.kind == "number" else spec.values[res["chosen"]]
            info: dict[str, Any] = {
                "requests": st.requests,
                "prompt_tokens_processed": st.prompt_tokens_processed,
                "per_request_prompt_n": st.per_request_prompt_n,
                "tokens_decoded": st.tokens_decoded,
                "legal_mass": float(f"{st.legal_mass or 0:.4g}"),
                "injected": st.injected,
                "decided_after": st.decided_after,
            }
            if st.accounted_mass is not None:
                info["accounted_mass"] = float(f"{st.accounted_mass:.6g}")
            if st.unexpanded_mass:
                info["unexpanded_mass"] = st.unexpanded_mass
            if st.prefix_prob is not None:
                info["prefix_prob"] = float(f"{st.prefix_prob:.4g}")
                info["prefix_checked"] = st.prefix_checked
            debug["per_question"][spec.key] = info
            tin += st.prompt_tokens_processed
            tout += st.tokens_decoded
        debug["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        debug["form_note"] = ("assembled from independent answers; the model never saw the whole form"
                              if mode == "independent" else "each answer conditioned on the earlier chosen answers")
        return LocalResponse(model=model or LOCAL_MODEL_NAME, usage=Usage(input_tokens=tin, output_tokens=tout), answers=answers, form=form, debug=debug)

    @staticmethod
    def _format(spec: _Spec, res: dict[str, Any]) -> Any:
        if spec.kind == "number":
            return {"type": "number", "value": res["value"], "probability": res["probability"]}
        p = res["probs"]
        if spec.jev == "noul":
            return NoulAnswer(noul=float(p[0]))
        if spec.jev == "score":
            n = len(p)
            mode = max(range(n), key=p.__getitem__)
            if n == 1:
                conf = 1.0
            else:
                center = (n - 1) / 2
                mad = sum(abs(i - center) for i in range(n)) / n
                conf = max(0.0, 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / mad)
            return ScoreAnswer(score=float(sum(i * pi for i, pi in enumerate(p))), confidence=conf,
                               probabilities={i: float(pi) for i, pi in enumerate(p)},
                               legend={i: c for i, c in enumerate(spec.source.model_dump(mode="json")["criteria"])})
        if spec.jev == "choice":
            named = {name: float(pi) for name, pi in zip(spec.values, p)}
            u = 1.0 / len(p)
            conf = 1.0 if len(p) == 1 else max(0.0, (max(p) - u) / (1 - u))
            return ChoiceAnswer(choice=max(named, key=named.__getitem__), confidence=conf, probabilities=named)
        best = max(range(len(p)), key=p.__getitem__)
        return {"type": spec.kind, "value": spec.values[best], "probabilities": [{"value": v, "p": float(pi)} for v, pi in zip(spec.values, p)]}


# --------------------------------------------------------------------------- serve


def _send(h: BaseHTTPRequestHandler, status: int, payload: Any, headers: dict[str, str] | None = None) -> None:
    body = json.dumps(payload).encode()
    h.send_response(status)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(body)))
    for k, v in (headers or {}).items():
        h.send_header(k, v)
    h.end_headers()
    h.wfile.write(body)


def _err(h: BaseHTTPRequestHandler, status: int, msg: str) -> None:
    _send(h, status, {"error": {"type": "error", "message": msg}})


def build_handler(jev: JevLocal, key: str | None, verbose: bool) -> type[BaseHTTPRequestHandler]:
    import threading

    lock = threading.Lock()  # one slot on the server: serialise

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            if verbose:
                super().log_message(fmt, *args)

        def _ok(self) -> bool:
            if not key or self.headers.get("Authorization", "") == f"Bearer {key}":
                return True
            _err(self, 401, "Invalid API key.")
            return False

        def do_GET(self) -> None:  # noqa: N802
            if not self._ok():
                return
            p = self.path.rstrip("/")
            if p in ("/v1/models", "/models"):
                _send(self, 200, {"models": [{"name": LOCAL_MODEL_NAME, "description": "Local System One over llama-server token probabilities.", "release_date": "2026-09-26"}]})
            elif p in ("", "/health"):
                _send(self, 200, {"ok": True})
            else:
                _err(self, 404, f"Unknown path {self.path}")

        def do_POST(self) -> None:  # noqa: N802
            try:
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            except ValueError:
                raw, self.close_connection = b"", True
            if not self._ok():
                return
            if self.path.rstrip("/") not in ("/v1/systemone", "/systemone"):
                _err(self, 404, f"Unknown path {self.path}")
                return
            try:
                req = json.loads(raw or b"{}")
                if not isinstance(req, dict):
                    raise ValueError
            except ValueError:
                _err(self, 400, "Body must be a JSON object.")
                return
            if "state" not in req or not req.get("questions"):
                _err(self, 422, 'Body needs "state" and a non-empty "questions" object.')
                return
            try:
                with lock:
                    r = jev.system_one(req["state"], req["questions"], mode=req.get("mode", "independent"), model=req.get("model"))
            except TypeSafeError as e:
                _err(self, 422 if "not supported" in str(e) else 502, str(e))
                return
            except Exception as e:
                if verbose:
                    traceback.print_exc()
                _err(self, 422, f"{type(e).__name__}: {e}")
                return
            payload = r.model_dump(mode="json")
            if not req.get("debug"):
                payload.pop("debug", None)
            if verbose:
                print(f"[jev-local] {r.debug['mode']} {len(req['questions'])} q, {r.debug['latency_ms']:.0f} ms", flush=True)
            _send(self, 200, payload, {"x-typesafe-request-id": f"local-{int(time.time() * 1000)}"})

    return Handler


def serve(host: str, port: int, verbose: bool) -> None:
    jev = JevLocal()
    key = os.environ.get("JEV_SERVER_KEY")
    httpd = ThreadingHTTPServer((host, port), build_handler(jev, key, verbose))
    print(f"jev-local on http://{host}:{port} -> llama-server {jev.base_url} (auth={'on' if key else 'off'})")
    print(f"SDK clients: TYPESAFE_BASE_URL=http://<this-host>:{port}  TYPESAFE_API_KEY={key or 'anything'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def run_test() -> int:
    state = "Hi, I've been trying to connect my Stripe account for 3 days and it keeps failing. I'm losing sales. Please help ASAP."
    questions = {
        "is_urgent": Noul(instructions="The message conveys urgency or time-sensitivity"),
        "sentiment": Score(instructions="Customer sentiment", criteria=["very negative", "negative", "neutral", "positive"]),
        "team": Choice(instructions="Which team should handle this ticket?",
                       criteria={"billing": "payments, invoices, refunds", "integrations": "third-party connections", "account": "login and profile"}),
        "days_waiting": {"type": "number", "instructions": "How many days has the customer been trying?"},
    }
    jev = JevLocal()
    for mode in ("independent", "dependent"):
        try:
            r = jev.system_one(state, questions, mode=mode)
        except TypeSafeError as e:
            print(f"{mode}: FAILED: {e}")
            return 1
        print(f"=== {mode}")
        print(json.dumps(r.model_dump(mode="json"), indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Jev-style decisions from local token probabilities")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test")
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8100)
    s.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.cmd == "test":
        return run_test()
    serve(a.host, a.port, not a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
