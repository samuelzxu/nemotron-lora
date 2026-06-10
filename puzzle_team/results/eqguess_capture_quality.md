# equation_numeric_guess — Bedrock capture quality audit (distillation decision)

**TL;DR — DISTILL (opposite of the bit_manipulation conclusion).** Unlike bit_manipulation
(where existing data already saturated the category at 99.9% and captures added nothing), eq_guess
is *deeply* under-covered: the existing trained set contains only **21 distinct problems**, and
all 126 CoTs are the *same* "give up and output |a−b|" absdiff template. The Bedrock captures
solve **37 distinct problems exact-string** / **48 with numeric tolerance** of 136, **31 of which
the deterministic typographic solver cannot get**, and they contain **genuinely SOUND per-symbol
derivations** (digit-reversal transforms, modulo elimination, digit-wise ops) — not the "pure
guessing" the prior `eqguess_analysis.md` concluded. Recommend: ship the deterministic solver for
the easy `-`/`*`/`/` slice AND distill ~37 deduped sound capture traces (best-per-problem, pooled,
gpt-oss + DeepSeek led) for the hard slice. Combined static coverage rises to **52/136 = 38%
exact-string** (61/136 = 45% under tolerance) vs 30 solver-alone / 21 distinct in existing data.

---

## 0. Method

- Re-graded **every** row with the current `eval/grader.py` (`is_correct`, rel_tol=1e-2 for
  non-binary numerics). Reasoning was read from `answer_text` (CoT lives there; `reasoning_chars`
  is 0 for these models) and `text` (populated only when capture-time grader passed).
- Distinguished **exact-string** correctness (the honest official-metric signal) from
  **tolerance-only** hits (pred within 1% of gt but not equal — near-miss inflation).
- Prompts for all 136 problems recovered from `winner_repo/train.csv`; ran
  `puzzle_team/solvers/equation_guess_rule.py` over all 136 to measure solver↔capture
  complementarity.

---

## 1. Full-file validity (all 3264 rows/model = 136 problems × 24 gens, COMPLETE)

| model | stored-correct | re-graded correct (tol) | exact-string correct | tolerance-only (near-miss) | distinct probs (tol) | distinct probs (exact) |
|---|---:|---:|---:|---:|---:|---:|
| eq_deepseek | 211 | **233** | 198 | 26 | 30 | 24 |
| eq_gptoss   | 299 | **334** | 199 | **127** | 38 | 29 |
| eq_mistral  | 165 | **186** | 150 | 31 | 25 | 20 |

**Direction of the discrepancy: DEFLATION, not inflation.** Re-grading is *higher* than the stored
`correct` field for all three (e.g. gpt-oss 334 vs 299). The capture-time grader was stricter than
the current one. So stored labels *understate* tolerance-correctness.

**But beware tolerance near-misses.** gpt-oss has 127/334 (38%) tolerance-only "correct" — answers
off by a small amount that pass rel_tol=1e-2 but are *not* exact. These are NOT reliably
distillable (the reasoning reaches a slightly wrong number). The honest distillable signal is the
**exact-string** column. The recurring offender is problem `0f8452df` (gt=159): every model
derives digit-wise add `9+6=15 | 0+8=8 → 158`, off by 1 from 159 — looks sound but is subtly wrong,
and only "passes" via tolerance.

---

## 2. Per-model 30-trace samples (mix incl. correct gens)

| model | sample re-graded correct | sample stored correct | correct-sample out_tokens (min/med/max) |
|---|---:|---:|---:|
| eq_deepseek | 20/30 | 16/30 | 1059 / 1893 / 3219 |
| eq_gptoss   | 20/30 | 17/30 | 725 / 1912 / 8192 |
| eq_mistral  | 20/30 | 19/30 | 1345 / 3781 / 5808 |

(Samples were deliberately weighted ~20 correct / 10 wrong to read distillable reasoning. Stored <
re-grade in every sample — consistent with the full-file deflation.)

---

## 3. Token budget — all within 8192

| model | median | mean | p90 | max | frac > 8192 |
|---|---:|---:|---:|---:|---:|
| eq_deepseek | 2171 | 2148 | 3365 | 5812 | 0.000 |
| eq_gptoss   | 4424 | 4583 | 8192 | 8192 | 0.000 |
| eq_mistral  | 3266 | 3503 | 5918 | 8192 | 0.000 |

Every trace fits an 8192-token distillable budget. DeepSeek is the most compact (median ~2.1K).
gpt-oss runs longer (median ~4.4K, some pinned at the 8192 cap) but never exceeds it.

---

## 4. Reasoning quality — SOUND derivations exist (this overturns the prior verdict)

The prior `eqguess_analysis.md` concluded the corrects were "prior-driven guesses." That is too
harsh for these captures: many correct traces perform a **legitimate per-symbol consistency
derivation** — they crack each example symbol's transform, *verify it against every example*, then
assign the query symbol its analogous operation. This IS the typographic rule, derived rather than
hard-coded, and it solves problems the deterministic solver and the winner cannot.

### eq_deepseek — strongest sound reasoning (compact, verifies on examples)

- **`386b6d03`** (`79'97`, ans 671, `rule_unknown`): derives `[`=("reverse both, multiply, reverse
  product"), **validates by switching the operation for the `-` example**, then maps `'`→add:
  > "[ means: reverse both, multiply, reverse the product. − means: reverse both, subtract …
  > Maybe ' means: reverse both, add, reverse the sum. … 97+79=176, Reverse 176→671" → \boxed{671} ✓
  A coherent symbol→operation framework, fully consistent with all examples. The winner couldn't
  solve this (`rule_unknown`).

- **`078df00e`** (`56*02`, ans 1031): infers "reverse each, operate, **+1 for addition**, reverse
  result," verifies on *both* `+` examples AND the `-` example, then extends to `*`:
  > "Test 45+99: reversed 54+99=153, add1=154, reverse=451. Yes! Test 06+67: …=137, reverse=731.
  > Yes! … check subtraction: 15−50=−35, reverse −35 = −53. Works." → applies to `*` → \boxed{1031} ✓

### eq_gptoss — sound but verbose; honest elimination

- **`20f0fac9`** (`31%96`, ans 2976, `rule_unknown`, 6990 tok): cracks `]`=modulo and `*`=multiply,
  explores and *rejects* many wrong sub-hypotheses, then eliminates:
  > "Thus maybe there are three operations: modulo, addition, multiplication. So we guess % is
  > multiplication. … 31%96 = 31 × 96 = 2976" → \boxed{2976} ✓
  Sound elimination over the operation set; a `rule_unknown` problem the winner missed.

### eq_mistral — sound on clean cases, but weaker self-consistency

- **`0f8452df`** (`90:68`): correctly identifies `$`=digit-wise subtraction, `@`=multiply, infers
  `:`=digit-wise addition (→158). Crisp, but the true answer is 159 — passes only on tolerance.
- **`1b3beb8f`** (`87|22`, ans 108): the prose concludes `|`=addition → "87+22 = 109" and boxes a
  value that grades correct *only* because 109 vs 108 is within 1%. **The reasoning does not soundly
  land on the answer** — a tolerance artifact, NOT distillable as-is.
  > "without more examples, it's hard to decide … the most likely operation for | is addition" → 109

**Verdict:** DeepSeek > gpt-oss > Mistral on *soundness density*. DeepSeek and gpt-oss frequently
produce verifies-on-examples derivations; Mistral more often rationalizes to a near-answer that
only survives tolerance. Filter to **exact-string** correct to drop the rationalized near-misses.

---

## 5. Existing trained dataset (126 eq_guess rows = only 21 distinct problems)

- All 126 `generated_cot` regrade correct — but **126/126 use the identical absdiff template**:
  brute-force enumerate dozens of transforms, declare none fit, then:
  > "We recall that the question operator is not found in the examples. We will use the absolute
  > difference as the operator. … |74 − 36| = 38" → \boxed{38}
- **18 of the 21 distinct problems have answer == |a−b|**; the other 3 only coincide because their
  answers are tiny (`03`,`02`,`04`).
- Typical length ~10.8K chars (~2.7K tokens). Mechanically uniform.

**This dataset teaches exactly ONE strategy** (enumerate → give up → |a−b|). It cannot teach the
model to solve `*`, `%`/modulo, reverse-concat, digit-wise add/sub, or symbol-analogy reasoning —
precisely the cases the captures handle. The category is genuinely under-covered.

---

## 6. Rule discovery — captures EXTEND the typographic rule

- **Confirm:** the `-`→subtraction signal and `*`/`/` tendencies show up in capture reasoning.
- **Extend (the important part):** captures soundly solve symbols the deterministic solver *cannot*
  encode, via example-derived transforms — `]`/`%`→modulo & multiply, `[`/`'`→reverse-concat &
  reverse-op, `$`/`:`→digit-wise sub/add, `+`-family. These are **novel sound derivations on
  `rule_unknown` problems** the winner (13.2%) missed (e.g. `386b6d03`, `20f0fac9`).
- **Do they "systematically discover" a master rule? No** — there is no single global symbol→op
  map (consistent with the prior info-theoretic finding), and a chunk of "correct" is
  tolerance-luck (`0f8452df`). But on the subset where the examples *do* pin the transform and the
  query symbol's literal meaning applies, the derivations are legitimate and reusable.

---

## 7. Coverage & overlap (only 136 problems exist)

**Distinct problems solved (exact-string / numeric-tolerance):**

| model | exact | tolerance |
|---|---:|---:|
| eq_deepseek | 24 | 30 |
| eq_gptoss   | **29** | **38** |
| eq_mistral  | 20 | 25 |
| **union (3 models)** | **37** | **48** |
| all-three overlap | 14 | 17 |
| gpt-oss-only (tol) | — | 12 |
| mistral-only (tol) | — | 1 |

**Solver ↔ capture complementarity (over all 136):**

| set | distinct solved |
|---|---:|
| deterministic solver alone | 30 / 136 (22.1%) |
| captures union (tolerance) | 48 / 136 (35.3%) |
| captures union (exact) | 37 / 136 |
| **solver ∪ captures (tolerance)** | **61 / 136 = 44.9%** |
| **solver ∪ captures (exact)** | **52 / 136 = 38.2%** |
| captures solve, solver misses (tol) | **31** |
| captures solve, solver misses (exact) | **22** |
| solver solves, captures miss | 13 |

The 31 capture-only problems span exactly the symbols the solver can't reason about:
`+`×9, `*`×5, `%`×3, `[`×2, `(`×2, `-`×3, plus `|{':\&/`. **Solver and captures are strongly
complementary** — together nearly double the static coverage of either alone.

**Coverage vs existing training data:** captures cover all 136 problems; the existing trained set
covers only **21 distinct**. Of the 48 problems captures solve, **34 are entirely absent** from the
existing training data, and of the 14 that overlap, 12 are already solver-covered absdiff cases.
Essentially **all new coverage is net-new.**

---

## 8. Distillable yield estimate (exact-string, in-budget, sound)

| model | exact-correct in-budget traces | distinct probs w/ ≥1 sound in-budget trace |
|---|---:|---:|
| eq_deepseek | 198 | 24 |
| eq_gptoss   | 199 | 29 |
| eq_mistral  | 150 | 20 |
| **pooled (best/problem)** | — | **37 distinct** |

Best-trace-per-problem source split: gpt-oss 17, DeepSeek 15, Mistral 5. After dropping
tolerance-only near-misses and capping ≤8192 tokens, you have **~37 deduped sound traces covering
37 distinct problems**, plus hundreds of redundant gens if multiple CoTs/problem are wanted for
robustness. **34 of these cover problems with NO existing training coverage.**

---

## 9. Recommendation — DISTILL (and combine with the solver)

**Yes, distill — this is the inverse of bit_manipulation.** There, existing data saturated the
category (99.9%) so captures were redundant. Here the existing data covers 21/136 distinct problems
with a single absdiff template, the winner gets 13.2%, and the captures add genuinely new,
sound, in-budget reasoning on 34 net-new problems.

Concretely:

1. **Ship the deterministic solver** (`equation_guess_rule.py`, 30/136 = 22.1%) for the easy
   `-`/`*`/`/` slice — free, exact, no model needed.
2. **Distill ~37 deduped sound CoT traces** (best-per-problem, pooled across models), filtered to
   **exact-string correct** and ≤8192 tokens, prioritizing **gpt-oss (17) and DeepSeek (15)** as
   the trace sources (they have the highest soundness density and most distinct coverage). Use
   Mistral sparingly (5) — it more often rationalizes to tolerance near-misses.
3. **Drop the 0f8452df-style tolerance-only hits** from the distill set; they look sound but land
   on the wrong number.
4. Optionally include 2–3 redundant CoTs per problem (from the ~547 pooled exact-correct in-budget
   gens) for the harder problems to give the LoRA more than one worked example.

**Expected payoff:** solver + distilled CoT gives **52/136 = 38% exact-string** static coverage,
and trains the model on symbol-analogy + example-verification reasoning it currently *cannot* do
(the existing data only teaches absdiff). Even at the category's irreducible-guessing ceiling
(~29% peeking oracle), the marginal value here is high precisely because the bar is low: 34 net-new
sound problems on a category at 13.2% baseline with 21 distinct trained problems is a clear win.

**Bottom line: distill (gpt-oss + DeepSeek led, ~37 deduped sound traces), and pair with the
deterministic solver. Opposite call to bit_manipulation.**
