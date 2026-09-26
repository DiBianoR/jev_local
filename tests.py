"""jev-local v2 test suite. Runs against a live llama-server.

  python tests.py http://127.0.0.1:5020 [--brute-k 4000]

Checks (plan §6):
  1. exactness: every enum/boolean distribution equals an exhaustive brute-force expansion
     of the token tree (both modes), with the branch floor set to 0
  2. dependent mode cost: 1 token decoded per decision; after the first field every
     request prefills only the injected stretch
  3. independent mode rewind: on a long document every question after the first
     processes < 100 prompt tokens, still true after 40 questions
  4. branch expansion: expansion requests prefill <= 5 tokens (checkpoint within reach)
  5. numbers, string refusal, and the Jev-compatible server with the official SDK
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time

import jev_local as J

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def brute(jev: J.JevLocal, base: list, fixed: str, spec: J._Spec, term: str) -> list[float]:
    """Exact distribution: expand EVERY legal token at EVERY position to the end of each answer."""
    inject, text, targets = jev._plan(fixed, spec, term)
    base2 = base + list(inject)
    probs = [0.0] * len(spec.values)
    st = J._Stats()

    def node(prefix: str, ids: tuple, mass: float) -> None:
        if mass < 1e-13:
            return
        live = [t for t in targets if t[0].startswith(prefix)]
        done = [t for t in live if t[0] == prefix]
        if done:
            probs[done[0][1]] += mass
            return
        entry = jev._complete(base2 + list(ids), J._enum_grammar(live, prefix), st)
        legal = {k: p for k, p in jev._candidates(entry).items() if k[1] and any(t[0].startswith(prefix + k[1]) for t in live)}
        z = sum(legal.values())
        for (tid, tok), p in legal.items():
            node(prefix + tok, ids + (tid,), mass * p / z)

    node("", (), 1.0)
    tot = sum(probs)
    return [p / tot for p in probs]


def dist_of(spec: J._Spec, answer) -> list[float]:
    if isinstance(answer, dict):
        return [x["p"] for x in answer["probabilities"]]
    if spec.jev == "noul":
        return [answer.noul, 1 - answer.noul]
    if spec.jev == "score":
        return [answer.probabilities[i] for i in range(len(spec.values))]
    return [answer.probabilities[n] for n in spec.values]


def test_wording_is_the_adapters(url: str) -> None:
    print("0. prompt wording is TypeSafe's adapter's, verbatim")
    from system_one_adapter import _schema as AS
    from system_one_adapter._client import _DISCRETE_SYSTEM_PROMPT, _OUTPUT_SCHEMA_INSTRUCTION_TEMPLATE, _serialize_state_as_user_prompt
    from pydantic_core import to_json
    qs = {"is_urgent": J.Noul(instructions="The message conveys urgency"),
          "sentiment": J.Score(instructions="Customer sentiment", criteria=["very negative", "negative", "neutral"]),
          "team": J.Choice(instructions="Which team?", criteria={"billing": "payments", "integrations": None}),
          "blocked": J.Noul(instructions="Customer is blocked", criteria={"true": "cannot work at all", "false": "has a workaround"}),
          "bare": J.Noul()}
    theirs = AS.create_raw_output_schema(AS.create_llm_output_model(AS.convert_question_collection_to_validated_api_question_models(qs), "discrete"))
    ours = J.schema_for([J._spec(k, q) for k, q in qs.items()])
    check(to_json(ours) == to_json(theirs), "schema for Jev questions is byte-identical to the adapter's")
    jev = J.JevLocal(url)
    doc = "Line one <b>\nLine two"
    specs = [J._spec(k, q) for k, q in qs.items()]
    pieces = jev._base_pieces(doc, specs)
    text = "".join(p if isinstance(p, str) else jev._detokenize((p,)) for p in pieces)
    check(_DISCRETE_SYSTEM_PROMPT in text, "system prompt: adapter's discrete prompt")
    check(_serialize_state_as_user_prompt(doc) in text, "document block: adapter's serialization")
    check(_OUTPUT_SCHEMA_INSTRUCTION_TEMPLATE.format(schema=to_json(theirs).decode()) in text, "schema instruction: adapter's template with the adapter's schema")
    check(text.index(_serialize_state_as_user_prompt(doc)) < text.index("Return one JSON object"), "document comes before the question-specific text (for caching)")


def test_exactness(url: str, k: int) -> None:
    print("1. exactness vs brute force (branch floor 0)")
    J.MAX_EXPANSIONS = 10**6
    jev = J.JevLocal(url, top_logprobs=k, min_branch_mass=0.0, prefix_check="off")
    doc = "A customer reports the export button crashes in Safari and asks for a refund."
    qs = {
        "blocked": J.Noul(instructions="The customer is blocked"),
        "sev": J.Score(instructions="Severity", criteria=["none", "low", "high", "total"]),
        "area": J.Choice(instructions="Which area?", criteria={"infra": None, "info": None, "into": None, "rust": None}),
        "n": {"type": "integer", "minimum": 0, "maximum": 13, "instructions": "Pick a number"},
        "w": {"enum": ["tea", "ten", "tile", "sail", "sails"], "instructions": "Pick a word"},
    }
    specs = {k_: J._spec(k_, q) for k_, q in qs.items()}
    # independent: brute force each question from its own base
    r = jev.system_one(doc, qs, mode="independent")
    worst = 0.0
    for key, spec in specs.items():
        bf = brute(jev, jev._base_pieces(doc, [spec]), J.independent_fixed(spec), spec, J.ANSWERS_CLOSE)
        d = max(abs(a - b) for a, b in zip(dist_of(spec, r.answers[key]), bf))
        worst = max(worst, d)
    check(worst < 1e-9, f"independent: max |alg - brute| = {worst:.1e}")
    # dependent: capture each field's base by wrapping _decide_enum
    orig = jev._decide_enum
    diffs = []

    def wrapped(base, fixed, spec, term, stats, check_prefix=False):
        out = orig(base, fixed, spec, term, stats, check_prefix=check_prefix)
        bf = brute(jev, base, fixed, spec, term)
        diffs.append(max(abs(a - b) for a, b in zip(out[0], bf)))
        return out

    jev._decide_enum = wrapped
    jev.system_one(doc, qs, mode="dependent")
    check(max(diffs) < 1e-9, f"dependent: max |alg - brute| over {len(diffs)} fields = {max(diffs):.1e}")


def test_dependent_cost(url: str) -> None:
    print("2. dependent mode: only decision tokens are decoded")
    jev = J.JevLocal(url)
    qs = {}
    for i in range(10):
        qs[f"f{i}"] = [J.Noul(instructions=f"statement {i}"),
                       J.Choice(instructions=f"question {i}", criteria={"infra": None, "info": None, "rust": None}),
                       J.Score(instructions=f"level {i}", criteria=["a", "b", "c"])][i % 3]
    r = jev.system_one("The deploy failed twice and customers see 500s.", qs, mode="dependent")
    pq = r.debug["per_question"]
    decoded = sum(v["tokens_decoded"] for v in pq.values())
    requests = sum(v["requests"] for v in pq.values())
    later = [n for key in list(qs)[1:] for n in pq[key]["per_request_prompt_n"]]
    check(decoded == requests, f"tokens decoded ({decoded}) == requests ({requests}), i.e. 1 per decision")
    check(decoded <= 2 * len(qs), f"{decoded} decisions for {len(qs)} fields (<= 2 per field with these options)")
    check(max(later) <= 20, f"after the first field every request prefilled <= 20 tokens (max {max(later)})")
    check(all(pq[k]["accounted_mass"] > 0.99 for k in qs if "accounted_mass" in pq[k]), "accounted mass > 0.99 on every field")
    print(f"       form: {json.dumps(r.form)}")


def test_independent_rewind(url: str) -> None:
    print("3. independent mode: rewind to end of document (long document)")
    jev = J.JevLocal(url)
    random.seed(time.time_ns())  # fresh document each run, so the first question can't hit a stale cache
    words = "the customer said export button crashes safari stripe billing refund deploy outage login slow angry".split()
    doc = " ".join(random.choices(words, k=2600))
    qs40 = {f"r{i}": J.Noul(instructions=f"statement {i}") for i in range(40)}
    qs = {f"q{i}": (J.Noul(instructions=f"statement {i}") if i % 2 == 0 else
                    J.Choice(instructions=f"question {i}", criteria={"infra": None, "info": None, "rust": None})) for i in range(8)}
    r = jev.system_one(doc, qs, mode="independent")
    first = r.debug["per_question"]["q0"]["prompt_tokens_processed"]
    later = [r.debug["per_question"][k]["prompt_tokens_processed"] for k in list(qs)[1:]]

    def tail_tokens(key: str) -> int:
        """Tokens after the document (delimiter + schema instruction + assistant header + injected answer)."""
        spec = J._spec(key, qs[key] if key in qs else qs40[key])
        pieces = jev._base_pieces(doc, [spec])
        return len(jev._tokenize(J.DELIMITER)) + len(jev._tokenize(pieces[-1])) + len(jev._tokenize(J.independent_fixed(spec))) + 2

    tails = [tail_tokens(k) for k in list(qs)[1:]]
    check(first > 2000, f"first question read the document ({first} tokens)")
    check(all(n <= t for n, t in zip(later, tails)), f"questions 2-8 processed {later} prompt tokens: only their own tails ({tails}), not the document")
    r = jev.system_one(doc, qs40, mode="independent")
    later = [r.debug["per_question"][k]["prompt_tokens_processed"] for k in qs40]
    over = [(k, n, tail_tokens(k)) for k, n in zip(qs40, later) if n > tail_tokens(k)]
    check(not over, f"40 questions on the same document: max {max(later)} prompt tokens per question, none beyond its own tail (checkpoint survived){' ' + str(over[:3]) if over else ''}")
    # dependent form on the same document, then an independent question: does the doc checkpoint survive?
    jev.system_one(doc, {f"d{i}": J.Noul(instructions=f"s{i}") for i in range(20)}, mode="dependent")
    r = jev.system_one(doc, {"after": J.Noul(instructions="afterwards")}, mode="independent")
    n = r.debug["per_question"]["after"]["prompt_tokens_processed"]
    qs40["after"] = J.Noul(instructions="afterwards")
    t = tail_tokens("after")
    print(f"       independent question after a 20-field dependent form processed {n} prompt tokens (its tail: {t})"
          + ("" if n <= t else "  -> rewound to an older checkpoint: the document checkpoint was evicted (PLAN §4; --ctx-checkpoints)"))


def test_branch_expansion(url: str) -> None:
    print("4. branch expansion rewinds cheaply")
    jev = J.JevLocal(url, min_branch_mass=0.0, prefix_check="off")  # only decision/expansion requests in the list
    J.MAX_EXPANSIONS = 10**6
    qs = {"w": {"enum": ["very negative", "very positive", "neutral", "negative"]}}
    r = jev.system_one("The customer is furious about the refund.", qs, mode="independent")
    pq = r.debug["per_question"]["w"]
    exp = pq["per_request_prompt_n"][1:]
    check(pq["requests"] >= 2, f"{pq['requests']} requests (expansions happened)")
    check(not exp or max(exp) <= 5, f"expansion requests prefilled {exp} tokens (<= 5 each)")
    check(abs(sum(x['p'] for x in r.answers['w']['probabilities']) - 1) < 1e-9, "probabilities sum to 1")


def test_misc_and_server(url: str) -> None:
    print("5. numbers, string refusal, server + official SDK")
    jev = J.JevLocal(url)
    r = jev.system_one("It happened 3 times.", {"n": {"type": "number", "instructions": "How many times?"}, "b": {"type": "boolean"}}, mode="dependent")
    check(isinstance(r.answers["n"]["value"], (int, float)) and 0 < r.answers["n"]["probability"] <= 1, f"number answer {r.answers['n']}")
    try:
        jev.system_one("x", {"s": {"type": "string"}})
        check(False, "string question refused")
    except J.TypeSafeError as e:
        check("not supported" in str(e), f"string question refused: {e}")
    env = dict(os.environ, LLAMA_URL=url, JEV_SERVER_KEY="k123")
    proc = subprocess.Popen([sys.executable, "jev_local.py", "serve", "--port", "8101", "--quiet"], env=env, cwd=os.path.dirname(os.path.abspath(__file__)))
    try:
        time.sleep(1.5)
        os.environ["TYPESAFE_API_KEY"] = "k123"
        os.environ["TYPESAFE_BASE_URL"] = "http://127.0.0.1:8101"
        from typesafe_sdk import Choice, Noul, Score, TypeSafeAuthenticationError, TypeSafeClient
        c = TypeSafeClient()
        check([m.name for m in c.models.list().models] == ["jev-local"], "GET /v1/models")
        x = c.system_one(state="deploy failed, 500s", questions={"u": Noul(instructions="urgent"), "s": Score(criteria=["low", "mid", "high"]), "r": Choice(criteria={"infra": None, "info": None})})
        check(type(x).__name__ == "SystemOneResponse" and 0 <= x.answers["u"].noul <= 1 and x.answers["r"].choice in ("infra", "info"), "official typesafe-sdk round trip")
        try:
            TypeSafeClient(api_key="wrong").system_one(state="x", questions={"a": Noul()})
            check(False, "bad key rejected")
        except TypeSafeAuthenticationError:
            check(True, "bad key rejected (401)")
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:8101/v1/systemone", json.dumps({"state": "x", "mode": "dependent", "debug": True,
                                     "questions": {"a": {"type": "boolean"}, "b": {"enum": ["tea", "ten"]}}}).encode(),
                                     {"Content-Type": "application/json", "Authorization": "Bearer k123"})
        d = json.loads(urllib.request.urlopen(req).read())
        check(d["debug"]["mode"] == "dependent" and set(d["form"]) == {"a", "b"}, "HTTP dependent mode with debug + form")
        try:
            from langchain_typesafe import Noul as LN, TypeSafeClassifier
            import warnings
            warnings.filterwarnings("ignore")
            v = TypeSafeClassifier().invoke({"state": "x", "questions": {"u": LN(instructions="urgent")}}).nouls["u"].noul
            check(0 <= v <= 1, "langchain-typesafe round trip")
        except ImportError:
            print("  skip langchain-typesafe not installed")
    finally:
        proc.terminate()


def test_prefix_prob(url: str) -> None:
    print("6. prefix_prob vs a full-vocabulary readout, one position at a time")
    doc = "The deploy failed twice and customers see 500s."
    qs = {"is_urgent": J.Noul(instructions="Needs attention now"),
          "team": J.Choice(instructions="Which team?", criteria={"infra": None, "billing": None})}
    vocab = len(J.JevLocal(url)._post("/tokenize", {"content": "x"})["tokens"]) and 160000
    for mode in ("first", "full"):
        jev = J.JevLocal(url, prefix_check=mode)
        r = jev.system_one(doc, qs, mode="independent")
        for key, q in qs.items():
            spec = J._spec(key, q)
            info = r.debug["per_question"][key]
            inject, _, _ = jev._plan(J.independent_fixed(spec), spec, J.ANSWERS_CLOSE)
            n = 1 if mode == "first" else len(inject)
            base = jev._base_pieces(doc, [spec])
            ref = 1.0
            for i in range(n):  # ask for the WHOLE vocabulary's probabilities at each position
                d = jev._post("/completion", {"prompt": base + list(inject[:i]), "n_predict": 1, "n_probs": vocab,
                                              "post_sampling_probs": False, "top_k": 1, "cache_prompt": True})
                top = d["completion_probabilities"][0]["top_logprobs"]
                ref *= sum(math.exp(c["logprob"]) for c in top if c["id"] == inject[i])
            rel = abs(info["prefix_prob"] - ref) / max(ref, 1e-300)
            check(rel < 1e-3, f"[{mode}] {key}: prefix_prob {info['prefix_prob']:.4g} vs full-vocab readout {ref:.4g} ({info['prefix_checked']!r})")


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LLAMA_URL", "http://127.0.0.1:5005")
    k = int(sys.argv[sys.argv.index("--brute-k") + 1]) if "--brute-k" in sys.argv else 4000
    print(f"llama-server: {url}")
    test_wording_is_the_adapters(url)
    test_exactness(url, k)
    test_dependent_cost(url)
    test_independent_rewind(url)
    test_branch_expansion(url)
    test_misc_and_server(url)
    test_prefix_prob(url)
    print("\nALL PASSED" if not FAILS else f"\n{len(FAILS)} FAILED:\n  " + "\n  ".join(FAILS))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
