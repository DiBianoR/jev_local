# jev-local v2

(Naming: **jev-local** is this tool; **Jev** means TypeSafe's hosted model.)

Jev-style typed decisions (state + questions → answers with probabilities) on your own GPU, from the model's **real token probabilities**, on **stock llama-server**. No fork, no compiling.

## Prompt wording

Every word the model reads comes from **TypeSafe's own LLM adapter** (`system-one-adapter` 0.2.1, MIT), in its discrete-answer mode (the model returns one allowed value per question), imported from the package so it can't drift:

* system prompt: the adapter's `_DISCRETE_SYSTEM_PROMPT` ("Evaluate every question using only the supplied document. … Return exactly one allowed value for each question.")
* user message: the adapter's document block (`<document>` + the JSON-encoded document with `<`/`>` escaped + `</document>`), then the adapter's schema instruction ("Return one JSON object that matches this schema exactly: … Do not include text or Markdown fencing before or after the JSON object.") with the adapter's schema: `{"answers": {…}}`, one property per question, descriptions like "Choice labels, answer with one label:". `tests.py` checks it is byte-identical to the adapter's.

One deviation: the adapter puts the schema instruction in the system prompt, before the document. jev-local puts the same text right after the document so the document can stay cached while the question changes. Native (non-Jev) question types get a property in the same style: the caller's description (or the adapter's "No additional instructions.") plus standard JSON-Schema type keywords.

## What it does

Two modes, both with the same guarantee: **the model only ever computes decision tokens.** Every other token (JSON keys, quotes, commas, the parts of an answer all options share) is injected into the prompt, never decoded, and decoding stops the moment the answer is unambiguous.

| | `independent` (default) | `dependent` |
|---|---|---|
| what the model sees | document + the schema for one question; answers `{"answers": {"<field name>": <value>}}` | document + the whole form; each field's answer is conditioned on the earlier chosen answers |
| cost per question / field | its own ~30–60 prompt tokens + 1 forward pass per decision | the injected stretch (≈6–10 tokens) + 1 forward pass per decision |
| document | read once; the server rewinds to its end between questions | read once |
| output | Jev-style answers + an assembled `form` (the model never saw the whole form) | Jev-style answers + the `form` as generated |

A decision costs one forward pass regardless of how much fixed text precedes it, because the request injects that text as prompt and asks for one token (`n_predict=1`).

**Measured on the test model** (hybrid architecture, 2,700-token document): first question ~2,800 prompt tokens; questions 2–40: 84–125 each (only the part of the schema that is specific to that question), never the document or the shared schema text again. Dependent 10-field form: 1 token decoded per decision, later requests prefilling 6–9 tokens.

### Answer types

| type | JSON | Jev primitive | returns |
|---|---|---|---|
| boolean | `true`/`false` | Noul | P(true) |
| enum of strings / ints / bools | the value | Choice (strings), Score (levels 0..n-1) | probability of every option |
| integer with min/max (≤200 values) | the value | – | probability of every value |
| number | JSON number (≤15 integer digits, ≤6 decimals) | – | value + probability of its token path; each digit is a decision |
| string | – | – | **refused** |

Native questions are JSON-schema-like dicts: `{"type": "boolean"}`, `{"enum": [...]}`, `{"type": "integer", "minimum": 0, "maximum": 10}`, `{"type": "number"}`, each with optional `"instructions"`.

### Probabilities

At each decision the raw-logit distribution (before temperature, penalties, grammar) is restricted to legal tokens by character matching against the remaining answers and renormalised. Alternative tokenisations of the same answer are summed. Alternatives that are still ambiguous after a token (`very` → "very negative"/"very positive") are expanded from their exact token-ID path. **Checked against exhaustive brute force: max difference ~1e-13** in both modes (`JEV_MIN_BRANCH_MASS=0`). With the default floor of 1e-4, branches carrying less mass than that are skipped and reported in `debug…unexpanded_mass`.

These are real model probabilities but not Jev-calibrated. Expect over-confidence; temperature scaling on a few hundred labelled examples is the standard fix.

## Setup

1. Get the official prebuilt llama.cpp for Windows + CUDA (build from **2026-09-23 or later**; the `message_delimiters` request field this relies on is recent).
2. Edit the four settings at the top of `start-llama-server.ps1` (llama.cpp folder, model path, context size, port) and run it. It mirrors textgen's llama.cpp loader flags and adds two that matter here: `--parallel 1` and `--ctx-checkpoints 64`.
3. Install and smoke-test:
   ```powershell
   cd C:\jev-local
   py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python jev_local.py test        # both modes on TypeSafe's support-ticket example
   python tests.py                 # the full suite against your running server (takes a minute)
   ```

## Use

```python
from jev_local import JevLocal, Noul, Score, Choice

jev = JevLocal()                                   # LLAMA_URL, default http://127.0.0.1:5005
r = jev.system_one(
    state="Deploy failed twice, customers see 500s. Can someone look now?",
    questions={
        "urgent":   Noul(instructions="Needs attention right now"),
        "severity": Score(instructions="Severity", criteria=["cosmetic", "degraded", "blocking"]),
        "route":    Choice(criteria={"infra": "servers/deploys", "billing": None, "frontend": None}),
        "retries":  {"type": "integer", "minimum": 0, "maximum": 10, "instructions": "How many failed deploys?"},
    },
    mode="independent",                            # or "dependent"
)
r.answers["urgent"].noul, r.answers["route"].probabilities, r.answers["retries"]["probabilities"]
r.form                                             # {"urgent": true, "severity": 2, "route": "infra", "retries": 2}
r.debug["per_question"]["route"]                   # requests, per_request_prompt_n, tokens_decoded, legal_mass, injected, ...
```

**Jev-compatible server:** `python jev_local.py serve --port 8100` (set `JEV_SERVER_KEY` for Bearer auth). Point clients at it with `TYPESAFE_BASE_URL=http://<server>:8100` and `TYPESAFE_API_KEY=<key>`; the official `typesafe-sdk` and `langchain-typesafe` work unchanged. Extra request fields: `"mode": "dependent"`, `"debug": true`. The response carries `form` in addition to Jev's `answers`.

## Diagnostics (`debug.per_question[key]`)

* `per_request_prompt_n`: prompt tokens the server actually processed per request. In independent mode the first question's first request should be about the document length and everything else < 100; in dependent mode requests after the first field should be a handful of tokens.
* `tokens_decoded`: should equal `requests` (one decision per request).
* `legal_mass`: raw probability on legal tokens at the first decision, given everything injected before it. Close to 1 on a good model.
* `prefix_prob`: probability that the model, unconstrained, would itself have written the injected opening (`prefix_checked` shows which text): off by default; `JEV_PREFIX_CHECK=first` measures `{"`, `full` the whole `{"answers": {"is_urgent": `. Low means the model was being pushed into a format it wasn't inclined to use. It's exact: the injected tokens are generated under a token-level grammar and llama-server reports their raw probabilities, even far outside the top-k. Cost: `first` adds about one forward pass per checked question (the injected tokens get their own small pass), `full` about one per injected token. Independent mode checks every question; dependent mode checks the form opening (every field's stretch with `full`). It's a diagnostic: turn it on to compare prompt wording, leave it off for speed.
* `injected` / `decided_after`: what was injected and the text at which the answer became certain.
* `timings`: per request, llama-server's `prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms`, plus `wall_ms` measured by jev-local (includes HTTP, sampling, grammar and checkpoint copies). `server_ms` / `wall_ms` per question are the sums. The gap between wall and server time is the per-request overhead.
* `checkpoint`: where this question's checkpoint was placed (`divergence point`, or the `document boundary` fallback).
* `accounted_mass`, `unexpanded_mass`: see Probabilities above.

## How the hybrid-model rewind works

Qwen3.8's recurrent layers can only be rewound to a saved checkpoint, and llama-server saves one immediately *before* each declared `message_delimiters` match. Everything after the document is identical for every question up to the first property name (56 tokens of schema text), so jev-local declares, per request, that question's first property name plus `":{"` (e.g. `is_urgent":{"`) as the delimiter. The checkpoint lands exactly where questions start to differ, and the next question rewinds to it. The text after the document is sent as its own natural token IDs, so the model sees exactly what it would see in a single message, tokenised normally. If that match isn't possible, jev-local falls back to a delimiter at the start of the schema instruction (the document boundary; each question then also re-reads the 56 shared tokens). `debug.per_question[key].checkpoint` shows which was used. Each decision request also leaves a checkpoint a few tokens before its end, which is what branch expansion rewinds to.

Checkpoints are stored in host RAM, 32 per slot by default. A dependent form creates ~2 per field, so after ~15 fields the document checkpoint can be evicted and the *next* independent question on the same document costs one re-read of up to `ubatch + 4` (516) tokens. `--ctx-checkpoints 64` in the start script pushes that to ~30 fields. Each checkpoint of Qwen3.8-27B is ~160 MB (48 recurrent layers × 48 heads × 128 × 128 float32 state), so 64 checkpoints can take ~10 GB of RAM; lower it if RAM is tight.

## Settings

| env var | default | |
|---|---|---|
| `LLAMA_URL` | `http://127.0.0.1:5005` | llama-server |
| `LLAMA_API_KEY` | none | if the server has `--api-key` |
| `JEV_TOP_LOGPROBS` | 50 | candidates read per decision |
| `JEV_MIN_BRANCH_MASS` | 1e-4 | `0` = exact |
| `JEV_PREFIX_CHECK` | `off` | `off` / `first` / `full`, see `prefix_prob` |
| `JEV_SERVER_KEY` | none | Bearer key for `serve` |

## Not done in this version (needs a llama-server fork)

* Explicit checkpoint positions (removes the delimiter text-matching).
* Prompt-token logprobs: would give `prefix_prob` for every injected token from the prefill itself, making `full` free.
* True simultaneous questions: decode every question's decision in the same forward passes over one shared document, making time nearly independent of the number of questions.

## Testing status

`tests.py` passes on two synthetic random-weight models with Qwen's tokenizer: a standard-attention one and a `qwen35` hybrid one (Gated-DeltaNet + attention, the same architecture family as Qwen3.8-27B), against real llama-server. Not yet run on Qwen3.8-27B itself. First thing to look at there: `legal_mass` (should be near 1) and `per_request_prompt_n` (should match the numbers above).
