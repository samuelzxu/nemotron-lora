# Bit-Manipulation Capture Quality Audit v2

**Scope:** Re-grade and quality-assess `bit_manipulation` reasoning traces from 5 Bedrock
captures (`bedrock/bitman_capture/bm_{deepseek,glm,gptoss,mistral,qwen}/results.jsonl`) vs the
existing trained-on dataset (`foundation/datasets/dgxchen_nemotron-cot-tong/problem_ids_matched.csv`,
`type=="bit_manipulation"`, 1754 rows). All grading uses the **current FIXED grader**
(`eval/grader.py`: binary `[01]+` answers compared **string-exact, no numeric tolerance**).
Decision: are Bedrock bit_manip traces worth distilling into the next LoRA set?

This builds on / reconciles with `puzzle_team/results/bitman_capture_audit.md` (v1).

---

## BOTTOM LINE

**Do NOT distill the Bedrock bit_manipulation captures. The existing trained-on dataset is
dramatically better — more valid, more sound, more distillable — and there is no meaningful
coverage gap that the captures would fill.**

- Bedrock stored-`correct` flags are **~63% inflated**: of stored-correct rows, only **~37%**
  are actually bit-exact under the fixed string-exact grader (deepseek 36.5%, glm 36.6%,
  gptoss 37.6%, mistral 38.6%, qwen 29.0%). The rest were bit-flipped near-misses the old
  decimal-tolerance grader accepted.
- Even the bit-exact Bedrock traces are **mostly unsound**: the dominant pattern is
  pattern-guessing / capitulation ("given the time, I'll guess…", "most similar example",
  "plausible") that happens to land on the right string. A large share of the bit-exact hits
  are the **degenerate all-zeros answer** (`00000000`), where almost any "output 0 unless
  special case" rationalization is correct by construction.
- The realistic ceiling of **sound, bit-exact, NON-degenerate** distinct problems per model is
  small: ~27–33 distinct problems for the four usable models, and fewer than half of those
  traces are genuinely sound on reading. Estimated genuinely-distillable traces across the
  best two models: **roughly 30–60 traces total** after hand-filtering.
- **gptoss correction to v1:** v1 said gptoss has no reasoning. That is WRONG for the `text`
  field. gptoss `answer_text` is just `\boxed{...}` (median 16 chars), but the `text` field
  **does** contain a full chain-of-thought (median 3420 chars). However that CoT is low quality
  (terse, "running out of time" guessing, one AES-S-box hallucination), so gptoss is still a
  poor distillation source — just not for the reason v1 gave.
- The **existing dataset is 99.9% bit-exact** (1752/1754) and its `generated_cot` is a clean,
  systematic **per-bit rule search**: lay out all bit columns → test every operator against ALL
  examples → select the "Perfect match" per bit → apply to the query bit-by-bit. This is exactly
  the SOUND, distillable derivation we want, at ~2200 tokens. It is already as good or better than
  anything in the captures.

---

## 1. Data layout (correction to v1's assumptions)

Each capture row stores reasoning in the `text` field **only when the capture-time grader marked
it correct** (`text` non-null ⟺ `correct==True`). Otherwise `text` is null and the model's final
text lives in `answer_text`. For grading purposes `text` and `answer_text` produce identical
re-grade counts on the stored-correct rows (the boxed answer is the same).

| model | rows | text non-null (= stored correct) | out_tokens median / max | reasoning text source |
|---|---|---|---|---|
| deepseek | 3600 | 706 | 3730 / 8192 | `text` |
| glm | 3600 | 453 | 1994 / 8192 | `text` |
| gptoss | 3600 | 631 | 1282 / 3623 | `text` (CoT); `answer_text` = boxed only |
| mistral | 3600 | 609 | 5234 / 8192 | `text` |
| qwen | 2525 (partial) | 138 | 8192 / 8192 | `text` |

All sampling below draws 30 rows from the `text`-non-null pool (every model has ≥138, so a full
30-row sample is available for each). `reasoning_chars` is only populated for gptoss (median 3448);
it is 0 for the others, so out_tokens is the length proxy.

---

## 2. Re-grade: stored `correct` vs string-exact (the inflation)

Re-graded the **entire stored-correct pool** of each model with the fixed grader:

| model | stored correct | bit-EXACT (regrade) | inflation (tolerance-only FPs) | regrade rate |
|---|---|---|---|---|
| deepseek | 706 | **258** | 448 | 36.5% |
| glm | 453 | **166** | 287 | 36.6% |
| gptoss | 631 | **237** | 394 | 37.6% |
| mistral | 609 | **235** | 374 | 38.6% |
| qwen (partial) | 138 | **40** | 98 | 29.0% |

These bit-exact totals exactly match v1's table — the v1 numbers were already string-exact and
reproduce cleanly. So **~63% of every model's "correct" flags are wrong bit strings** accepted
only by the old decimal tolerance. Never trust the stored field.

### 30-trace sample re-grade (seed=42, from text-non-null pool)

| model | sample bit-exact (regrade) | stored-correct in sample | of bit-exact: all-zeros / non-zero |
|---|---|---|---|
| deepseek | **13 / 30** | 30 / 30 | 6 / 7 |
| glm | **13 / 30** | 30 / 30 | 3 / 10 |
| gptoss | **16 / 30** | 30 / 30 | 6 / 10 |
| mistral | **17 / 30** | 30 / 30 | 8 / 9 |
| qwen | **8 / 30** | 30 / 30 | 6 / 2 |

Every sampled row was stored as `correct=True`, but only 8–17 of 30 survive string-exact grading.
And of those survivors, a large fraction are the **degenerate `00000000`** answer.

---

## 3. Degenerate all-zeros inflation

Many problems' query answer is `00000000`. For these, a model can write almost any "output 0
unless <condition>" rationalization, conclude the query doesn't meet the condition, and emit
`00000000` — bit-exact, but the derivation is unfalsifiable for that query. Stripping all-zeros
answers from the **full-file** bit-exact pools:

| model | full-file bit-exact | non-zero-answer bit-exact | distinct non-zero problems |
|---|---|---|---|
| deepseek | 258 | 144 | **31** |
| glm | 166 | 92 | **28** |
| gptoss | 237 | 139 | **27** |
| mistral | 235 | 140 | **33** |
| qwen (partial) | 40 | 6 | **5** |

The "distinct non-zero problems" column is the realistic **diversity ceiling** of usable
captures per model: ~27–33 problems for the four full models, only 5 for partial qwen. Note the
dataset has only 150 distinct problems, so these pools overlap heavily across models.

---

## 4. Reasoning quality (read, not heuristic) — per model

Verdicts are from reading the conclusion of each bit-exact sampled trace. The recurring failure
mode across ALL Bedrock models is **terminal capitulation-guessing**: the model fails to find a
rule, then picks an answer by analogy/symmetry and boxes it. Bit-exactness is then luck.

### deepseek — MOSTLY UNSOUND (a few sound)
Of 13 bit-exact, ~3 genuinely derive+verify+apply; the rest guess.
- SOUND (`3456da40`, GT `00001111`): derives right-shift-by-2, checks every example, applies:
  > "Example 7: `00011100` → `00000111` … o5..o0 = 0 0 0 1 1 1. Matches. … Apply to `00111101` …
  > output: 0 0 0 0 1 1 1 1 … `\boxed{00001111}`"
- SOUND (`8631d7b6`, GT `00000000`): "Output bit 7 = b1 AND b0. Output bits 6–0 = 0 … b1=0,b0=1 →
  0 … `\boxed{00000000}`" (all-zeros, but the rule is actually verified).
- UNSOUND / LUCKY (`16db2c74`, GT `00000110`): "Given the examples … I'll pick a plausible one …
  **I'll guess 00000110.** … **Final guess:** `\boxed{00000110}`" — pure guess.
- UNSOUND (`b4ddb69d`, GT `00000000`): "So I'm stuck, but for `00000000`, any XOR combination
  gives 0. Thus final answer likely `00000000`." — degenerate, no rule.

### glm — BEST of the captures (still ~50/50)
Of 13 bit-exact, ~5–6 are sound; glm more often writes an explicit per-bit formula and verifies.
- SOUND (`eeac10f6`, GT `11011110`): derives per-bit XOR formulas and double-checks:
  > "$y_7 = b_7 \oplus b_6 \oplus b_5 = 1 … = 1$ … Simpler verification using the derived
  > formulas: $y_7=1, y_6=1 … y_0=0$ Output: 11011110. `\boxed{11011110}`"
- SOUND (`6cface63`, GT `10010000`): "Rotate Left 3 … Swap bits 2,3 … Ex 1 … Matches. Ex 2 …
  Correct. Input `00010010` … Result: `10010000`." — rule applied and checked on examples.
- UNSOUND (`288c7eca`, GT `01111111`): "It is **highly probable** that the transformation follows
  the same logic … `\boxed{01111111}`" — analogy, not derivation.
- UNSOUND (`36a33623`, GT `00000000`): "**not finding a clear, consistent bit manipulation rule**
  … it's plausible that `00000001` would also output `00000000`." — explicit give-up.

### gptoss — HAS CoT (v1 was wrong) but LOW QUALITY
Correction: `text` holds a real chain-of-thought (median 3420 chars, 631 traces). But the CoT is
terse scratch-work that usually ends in time-pressure guessing, and `answer_text` is just the box.
- SOUND-ish (`3456da40`, GT `00001111`): "00001011 correct. 10101000 >>2 = 00101010 correct …
  Thus rule is simple logical right shift by 2 … `\boxed{00001111}`" — actually derived.
- UNSOUND (`36a33623`, GT `00000000`): "**Given limited time**, maybe output for 00000001 is
  00000000? … I'll answer 00000000."
- UNSOUND (`2cf45d07`, GT `10101010`): "**Running out time** … Given difficulty, maybe answer is
  01010101? Guess? … I'll answer that. `\boxed{10101010}`"
- HALLUCINATION (`108e69ef`, in stored-correct, tolerance-only FP): `answer_text` claims "The
  given pairs correspond exactly to the AES S-box mapping" — false, and not even boxed. Toxic.

### mistral — UNSOUND-LEANING (longest traces)
Of 17 bit-exact, ~3 sound; mistral writes a lot but frequently concludes by guessing. Also has the
highest all-zeros share (8/17 bit-exact were `00000000`).
- SOUND (`3456da40`, GT `00001111`): "rotated right by two … (matches fourth example) … This rule
  **consistently matches all given examples**. Applying to `00111101` … `\boxed{00001111}`"
- UNSOUND (`92b8f52a`, GT `01010100`): "**After carefully analyzing** … the **most plausible**
  output for `10100110` is `\boxed{01010100}`" — plausibility, not a rule.
- UNSOUND (`288c7eca`, GT `01111111`): "After **struggling to find a consistent rule** … the most
  plausible output … is `01111111` … `\boxed{01111111}`."

### qwen — WORST usable yield (partial + truncation)
Median out_tokens = 8192 (hits the cap), only 138 stored-correct in 2525 rows, 40 bit-exact, and
just **6 non-zero-answer bit-exact across 5 distinct problems**. Most sampled bit-exact are the
same degenerate problem `8631d7b6` (`00000000`). On the plus side, when qwen does solve `8631d7b6`
it writes a clean verified rule:
- SOUND-on-degenerate (`8631d7b6`): "the rule is: If the last 4 bits are `1011`, output
  `10000000`; otherwise `00000000`. … 10101011 → ends `1011` → `10000000` ✅ … All others …
  `00000000` ✅ … target `11011101` last 4 bits `1101` ≠ `1011` → `00000000`."
But the non-degenerate yield (5 problems) is too small to matter, and 57% of qwen traces blow past
the 8192-token budget (v1), making them non-distillable anyway.

---

## 5. Length / budget

| model | out_tokens median / p90-ish | over ~7680 budget? | distillable length? |
|---|---|---|---|
| deepseek | 3730 / ~4500 | rarely | yes |
| glm | 1994 / ~5700 | a few | yes |
| gptoss | 1282 (CoT ~3420 chars) | no | length OK, content weak |
| mistral | 5234 / ~6700 | a few % | mostly yes |
| qwen | 8192 / 8192 | **majority hit cap** | no (truncated/looping) |

deepseek, glm, gptoss, mistral fit an 8192-token budget. qwen mostly does not.

---

## 6. Existing trained-on dataset — SOUND and distillable

`type=="bit_manipulation"` rows: **1754**. Re-grade of `generated_cot` vs `answer`:
**1752/1754 = 99.9% bit-exact** (all 1754 contain `\boxed{}`). Sample of 30: **30/30 bit-exact**.
Length: median 8786 chars ≈ **2197 tokens**, p90 ≈ 2337 tokens — comfortably within budget.

The `generated_cot` is a uniform, fully systematic per-bit rule search. Every trace:
1. Lays out all input and output bit columns across all 10 examples.
2. For each output bit, tests each operator term (Identity / NOT / Constant / AND / OR / XOR /
   AND-NOT / OR-NOT / XOR-NOT / shifts / rotates / pairwise like `XOR70`, `AND23`) against ALL
   examples, marking "Perfect match".
3. Selects the matching per-bit operation, then applies it to the query input bit-by-bit.

Excerpts:
> "Identity no NOT no Constant no AND no OR no XOR yes AND-NOT no … **Matched** 0 I2 1 I3 2 I4 3 I5
> 4 I6 5 XOR70 6 I1 7 I2 … **Applying to 11111101** … 5 XOR70 = XOR(1,1) = 0 … `\boxed{11110011}`"
> (id `7a5d00a7`, sound: per-bit ops verified then applied)

> "**Matched** 0 AND23 1 AND34 2 AND45 … **Applying to 00100000** … 0 AND23 = AND(1,0) = 0 …
> `\boxed{00000000}`" (id `58b650e5` — even the all-zeros case is reached by an explicitly
> verified per-bit AND rule, not a guess)

> "**Perfect match** … Matched 0 I7 1 I0 2 I1 … 7 I6 … Applying to 11000000 … 0 I7 = 0 1 I0 = 1 …
> `\boxed{01100000}`" (id `b14fb614`, a verified rotate)

There is **zero capitulation/guess language** in the sampled 30 (verify-language search hits the
structured "Perfect match" markers; guess-language search returns 0). The traces are mechanical and
templated (low natural-language variety) but they are exactly the derive→verify-on-all→apply pattern
we want a student model to learn, and they are correct.

---

## 7. Head-to-head and ranking

**Existing dataset vs Bedrock captures:** not close. The existing data is 99.9% bit-exact and
uniformly sound (explicit per-bit verification on all examples), at ~2200 tokens. The best Bedrock
model (glm) is ~37% bit-exact in its "correct" pool, and only roughly half of those bit-exact
traces are sound on reading, with many wins being the degenerate all-zeros case.

**Bedrock model ranking by usable-trace quality (best → worst):**
1. **glm** — best soundness rate (most likely to write+verify an explicit per-bit formula),
   shortest sound traces (~1.3–2k tokens), 28 distinct non-zero problems. Highest signal.
2. **deepseek** — clean SOUND traces exist (`3456da40`, `8631d7b6`) and fit budget (~3.7k tok),
   but capitulation-guessing dominates; 31 distinct non-zero problems.
3. **mistral** — 33 distinct non-zero problems and verbose derivations, but unsound-leaning
   (lots of "plausible"/"struggling") and highest all-zeros share; longer (~5k tok).
4. **gptoss** — does have CoT, but it's terse time-pressure scratch-work with a hallucination
   (AES S-box) and the polished final answer carries no derivation. Low yield.
5. **qwen** — worst: partial file, 57% over budget/truncated, only 5 distinct non-zero problems.

---

## 8. How many sound, distillable traces could Bedrock realistically yield?

Take the per-model non-zero-answer bit-exact distinct-problem ceilings (§3): deepseek 31, mistral
33, glm 28, gptoss 27, qwen 5. These pools overlap heavily (only 150 problems exist; v1 found 69
distinct bit-exact problems total across all models, ~39 solved by ≥2 models). On reading, only
about **40–55% of bit-exact non-degenerate traces are genuinely sound** (rest are
guess-rationalizations). So the union of genuinely-sound, bit-exact, non-degenerate, in-budget
problems across the best sources (glm + deepseek, with mistral as backup) is on the order of
**~30–60 distinct traces** — and essentially all of those problems are already covered, soundly,
by the 1754-row existing dataset.

---

## 9. Recommendation

**Do not distill the Bedrock bit_manipulation captures into the next LoRA training set.**

Reasons:
1. **Validity:** stored-correct is ~63% inflated; true bit-exact is ~37% (29% for qwen). Distilling
   the raw "correct" set would teach near-miss bit strings — the exact wrong lesson for a
   string-exact graded task.
2. **Soundness:** even bit-exact captures are dominated by capitulation-guessing and degenerate
   all-zeros wins. Realistic genuinely-sound yield is only **~30–60 distinct traces**, and that
   requires expensive per-trace hand-filtering plus a rule-verifier pass.
3. **No coverage gap:** the existing 1754-row dataset is 99.9% bit-exact and uniformly sound (the
   per-bit "Perfect match" search). It already covers the problem space at far higher quality than
   the captures could add. There is no subset of problems the captures solve soundly that the
   existing data does not already cover well.

If more bit_manip data is wanted, the right move is **not** to mine these captures but to
**generate fresh traces in the existing dataset's per-bit-search style** (programmatically
verifiable, string-exact graded), or simply up-weight the existing 1754 rows. gptoss and qwen
should be excluded outright. If one insists on harvesting captures, restrict to glm + deepseek,
re-grade string-exact, drop all-zeros answers, drop traces containing guess language
("guess"/"plausible"/"symmetry"/"running out of time"/"struggling"), and re-verify each surviving
rule with `puzzle_team/results/bitman_verify.py` — expect to keep only a few dozen, with no
material benefit over the existing data.
