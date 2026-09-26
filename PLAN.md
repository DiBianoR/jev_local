# jev-local v2 plan

Goal: Jev-style typed decisions on Qwen3.8-27B via stock llama-server (no fork), meeting:

1. **Dependent forms** (each answer sees the earlier ones): the model computes only decision tokens. Every token before and between decisions is injected as prompt.
2. **Independent questions** (each sees only the document): no tokens are computed before the decision; after each question, the server rewinds to the end of the document instead of re-reading it.
3. Both modes in one tool, with the existing Jev-compatible API kept.

Target server: stock `llama-server` from the official prebuilt Windows CUDA release, built from llama.cpp master of 2026-09-23 or later (the build whose source this plan was checked against). jev stays a separate Python program using HTTP only.

---

## 0. What stays from v1

- Question types: boolean, enum (strings / ints / bools), bounded integer (≤200 values, treated as enum), number. Free strings are refused.
- Probability accounting: at each decision step the raw-logit distribution (`post_sampling_probs=false`) is restricted to legal tokens by character matching and renormalised; alternative tokenisations of the same option are summed; ambiguous branches (`very` → "very negative" / "very positive") are expanded from their exact token-ID path. Verified against exhaustive brute force to ~1e-12 in v1.
- Jev-compatible `POST /v1/systemone` server, `typesafe-sdk` and `langchain-typesafe` compatibility.
- Chat template applied by the server (`/apply-template`, thinking off).

## 1. Prompt layout and the document snapshot

Every request is sent to `/completion` as an **array prompt** of separately tokenised pieces:

```
[ <template head + document block>,  <delimiter token IDs>,  <rest of schema instruction + assistant header>,  <injected answer text>,  <decision token IDs...> ]
```

with `"message_delimiters": [{"role": "user", "delimiter": "\n\nReturn one JSON object that matches this schema exactly:"}]` (the start of the adapter's own schema instruction; see §1b).

Why: llama-server saves a context checkpoint immediately before every occurrence of a declared delimiter's token sequence, and rewinds to the nearest checkpoint at or before the point where a new prompt diverges. On a hybrid model (Qwen3.8 = 3 Gated-DeltaNet layers : 1 attention layer, loaded as `qwen35`) that is the only way to rewind: per-token KV entries can be cut anywhere, but the recurrent state can only be restored from a checkpoint. Declaring the end-of-document boundary as a delimiter puts a checkpoint exactly there. This piece is sent as its own array element so its tokens are guaranteed identical to the delimiter's tokens (both go through `common_tokenize(…, add_special=false, parse_special=true)` in the server; verified in `tokenize_mixed` and `common_chat_msg_delimiters::tokenize`).

The model sees exactly the same characters as v1. No extra tags, no extra message boundaries.

> **Choice made — delimiter text.** Options: (a) natural boundary text already in the prompt (picked; since §1b that text is the start of the adapter's schema instruction); (b) an artificial tag like `</document>`; (c) two user messages (rejected: changes what the model sees). (a) adds nothing to the prompt. Its only cost: if the same text occurs inside a document, each occurrence creates an extra, harmless checkpoint.

## 1b. Prompt wording (changed 2026-09-26)

All wording is TypeSafe's LLM adapter's (system-one-adapter 0.2.1, MIT), discrete-answer mode, imported from the package: system prompt, document block (JSON-encoded, `<`/`>` escaped), schema instruction template, and the output schema `{"answers": {...}}` with the adapter's per-question descriptions. `tests.py` checks byte-identity with the adapter's schema and text.

> **Choice made — where the schema instruction goes.** Options: (a) the adapter's placement, in the system prompt before the document (rejected: every question would re-read the document); (b) the same text directly after the document (picked). Only the position differs.
>
> **Choice made — spacing of the injected JSON.** The adapter doesn't specify output formatting. jev-local injects `{"answers": {"key": ` (a space after `:` and `,`). Compact `{"answers":{"key":` is the alternative. `prefix_prob` measures which the model prefers; switch if the real model scores compact higher.
>
> **Consequence on hybrid models:** the checkpoint sits at the delimiter, so each question re-reads the ~60 schema tokens every question shares before its own property. A standard-attention model rewinds to the exact divergence point. Moving the delimiter to the first property would need splitting the schema text; not worth it (same single prefill pass on a GPU).
>
> **Fork would do better:** a request field "checkpoint after token N" would remove the text-matching dependency entirely. Not worth a fork on its own; fold it into the option-C fork if that happens.

## 2. Dependent mode (one form, answers in sequence)

Output shape is a flat JSON object in the question order: `{"k1": v1, "k2": v2, …}`.

Loop per field:
1. **Inject** everything known up to the next decision: the remainder of the previous value after its decision token, `", "k2": `, and any opening tokens all of k2's options share (e.g. `"` for string enums, `"very ` if all options start with it). Sent as a string piece → tokenised by the server.
2. **One request, `n_predict=1`**: the server prefills the injected stretch and returns the probabilities at the decision position in the same forward pass. Cost ≈ one decode step regardless of the injected length (weight-read bound on the A6000).
3. Read probabilities, choose the best legal token (greedy), record it as a token ID.
4. If the field is still ambiguous after that token (`in` → infra/info), repeat 2–3 with the chosen ID appended. Otherwise the field is decided; go to the next field.
5. After the last field's decision: **stop**. The closing text is assembled client-side; nothing more is computed. (If a later request continues from the completed form, the tail gets prefilled then, at no extra cost now.)

Per-field probabilities are conditional on the earlier fields' *chosen* values (inherent to dependent forms; documented in the response).

Numbers: every digit and the terminator (`,` / `}`) are decisions; each digit costs one step. Reported: the value plus the probability of its exact token path.

> **Choice made — how to tokenise injected text.** Options: (a) tokenise each injected stretch on its own, independently of the decision token before it (picked); (b) let the model generate the fixed stretch under the grammar (v1 behaviour: costs a pass per token, defeats the purpose); (c) re-tokenise the whole answer-so-far and splice (would change the decision tokens whose probabilities were already read). (a) can produce a token boundary the model would not have chosen itself (the classic "token healing" concern); the injected text is always punctuation/keys/shared prefixes, so the effect is small. Measured impact is a test item (§6).
>
> **Backtrack path:** if (a) measurably hurts, fall back to (b) for the first token after a decision only.

Branch expansion (ambiguous alternatives not taken) in dependent mode requires rewinding within the form. That depends on near-end checkpoints (§4). Default in dependent mode: expand only branches whose mass ≥ `JEV_MIN_BRANCH_MASS` (1e-4), same as v1, so most fields need no rewind at all.

## 3. Independent mode (each question alone)

Per question, one request whose prompt is: document piece, delimiter piece, then the schema instruction for that question + assistant header + `{"answers": {"<key>": ` + shared opening, then decision IDs — same stepwise loop as §2 but with a single field. Between questions, the server rewinds to the document checkpoint and prefills only the question's own tail (~90–180 tokens with the adapter's schema text).

Questions run **sequentially on one slot** (`--parallel 1`). Running them on several slots re-reads the document once per slot (measured in v1: 4 slots → ~4× prompt tokens; `--kv-unified` does not help).

> **Fork would do better (option C, deferred):** a batched endpoint that keeps the document once and decodes all questions' decision tokens in the same forward passes. This makes time nearly independent of the number of questions. Requires a llama-server patch and rebuilding on every update. Not in this version.

Response also carries an assembled `form` object `{key: value}` built from the separate answers, as a convenience. It is labelled as assembled; the model never saw the whole form.

## 4. Checkpoint budget (hybrid models only)

Facts from the source (2026-09-23):
- Max checkpoints per slot: `--ctx-checkpoints` (default 32). Each stores the recurrent state; size is printed in the server log.
- Created: at every declared delimiter position (always, for the last user delimiter), and 4 tokens and 4+`n_ubatch` tokens before the end of every prompt. So each decision step adds up to 2 near-end checkpoints.
- When the list is full, checkpoints closer than `--checkpoint-min-step` (default 8192) to an earlier one are evicted first, then the oldest.

Consequences to verify by test (§6):
- The document checkpoint must survive a long independent-question run. Each question's request re-hits the delimiter position, which re-creates/supersedes the checkpoint there, so it should.
- In dependent mode the document checkpoint is not re-hit (the prompt only grows). After ~16 decision steps it may be evicted. That only matters if something later rewinds past the form (the next independent question on the same document) — cost: one document re-read.
- Branch expansion rewinds to a previous decision position, which lies within 4 tokens of the end of the request that produced it, so the "4 before end" checkpoint of that request covers it — if it has not been evicted.

> **Choice made — checkpoint pressure.** Options: (a) rely on defaults and measure (picked first); (b) raise `--ctx-checkpoints` in the start script if measurements show eviction and the per-checkpoint size is small enough; (c) in dependent mode, re-hit the document delimiter by issuing a no-op request before the next independent question (forces one recreate); (d) fork: explicit checkpoint positions. Start with (a); (b)/(c) are one-line changes.

## 5. Server start script (`start-llama-server.ps1`)

Replaces textgen's llama-server for this model. Mirrors the textgen setup:
- model path, `--ctx-size` (max context Robert uses), `--n-gpu-layers 999`, `--flash-attn on`
- MTP speculative decoding as textgen enables it (irrelevant to jev — every jev request decodes ≤1 token — but keeps normal chat use fast)
- `--parallel 1` (see §3), `--port 5005` (or as chosen), `--host 127.0.0.1` unless LAN access is wanted, `--api-key` optional
- `--ctx-checkpoints` left at default until §6 says otherwise
- Comments noting each flag's origin.

SillyTavern and other clients use this server's `/v1/chat/completions` directly.

## 6. Tests (pass criteria = the two goals)

Run against a real llama-server. Two synthetic models with Qwen's tokenizer:
- the existing random-weight `qwen2` model (standard attention), and
- a new tiny random-weight **`qwen35`** model built from llama.cpp's tensor spec, so the hybrid checkpoint path is actually exercised.

1. **Exactness** (both models): brute-force comparison as in v1, both modes, `JEV_MIN_BRANCH_MASS=0` → max |diff| ≤ 1e-9.
2. **Dependent mode cost**: for a 10-field form, tokens *decoded* == number of decision tokens; each request's `prompt_n` == injected stretch length; no request decodes more than 1 token.
3. **Independent mode rewind (hybrid)**: 2,700-token document, 8 questions → first question processes ~2,700 prompt tokens, every later one processes < 100. Same after 40 questions (checkpoint survival).
4. **Branch expansion (hybrid)**: an enum with shared prefixes; expansion requests process ≤ 5 prompt tokens each (checkpoint within 4 tokens).
5. **Injected-tokenisation effect** (§2 choice): on the random model this is unmeasurable; note it as a real-model check: compare P(answer) for a field decided directly vs. after an injected stretch, over 50 documents on Qwen3.8-27B, expect no systematic shift. This one runs on Robert's machine.
6. SDK / LangChain / auth / error-path tests from v1.

Deliverables: `jev_local.py`, `start-llama-server.ps1`, `README.md`, `requirements.txt`, test script.

## 6b. Test results (2026-09-26)

All of §6 items 1-4 and 6 pass on both synthetic models (standard-attention `qwen2` and hybrid `qwen35`), see `tests.py`:
- exactness: max |alg - brute| 1.5e-13 (independent), 1.1e-13 (dependent), floor 0
- dependent 10-field form: decoded == requests; later requests prefill 6-9 tokens
- independent, 2,700-token document: first question 2,695 tokens, questions 2-40: 34-46 tokens
- branch expansion requests prefill 1 token
- §4 prediction confirmed: with the default 32 checkpoints, an independent question after a >=20-field
  dependent form re-reads 514 tokens (the ubatch+4 fallback checkpoint). `--ctx-checkpoints 128` fixed it
  through 40 fields; the start script uses 64. Checkpoints are host RAM.
- found during testing: `/completion` treats an all-string array prompt as several prompts; jev sends the
  delimiter as token IDs so the array is always mixed (one prompt). Also unbounded number grammars let a
  weak model emit digits forever; numbers are now capped at 15 integer + 6 fraction digits.
- §6 item 5 (injected-tokenisation effect on the real model) is still open; runs on Robert's machine.

## 7. Deferred / fork-only improvements (summary)

| Improvement | Needs fork? | Gain |
|---|---|---|
| Explicit checkpoint positions | yes | removes delimiter text-matching, frees checkpoint budget |
| Option C: batched multi-question decode over one shared document | yes | independent-mode time ~independent of question count |
| Prompt-token logprobs (logits at several prompt positions in one pass) | yes | `prefix_prob` for the whole injected opening at no extra decode cost (today `full` costs 1 step per token) |
