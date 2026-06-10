# Bit-Manipulation Capture Audit

**Scope:** 5 capture files `bedrock/bitman_capture/bm_{deepseek,glm,gptoss,mistral,qwen}/results.jsonl`
(16,773 records total; **2,531 flagged `correct=True`** with reasoning text).
**Method:** exact bit re-verification, grader reconstruction, brute-force rule verifier
(`puzzle_team/results/bitman_verify.py`), token/degeneracy scans, cross-model agreement,
and manual reading of representative traces. Tools: `foundation/.venv/bin/python`, `eval/grader.py`.

---

## BOTTOM LINE

**This corpus is NOT trustworthy to distill as-is. It is heavily grader-contaminated.**

- Only **36.9% (935/2,531)** of "correct" records are **bit-exact**. The other **63.1% (1,596)**
  are accepted *only* by the metric's decimal numerical tolerance and are **wrong bit strings**
  (Hamming distance 1–5 from ground truth).
- Even among bit-exact records, **most are lucky guesses / backward-rationalizations**, not derived rules.
  Hand-validated genuinely-SOUND traces (derive rule → verify on all examples → apply correctly) number on
  the order of **~15–25 distinct problems**, i.e. **roughly 0.5–1% of flagged-correct records**.
- The `bm_gptoss` slice is unusable for distillation: **619/631 traces have no reasoning text** (only `\boxed{..}`).
- The dataset ground truth itself is **sound** (independently verified), so the problem is the captures + metric, not the data.

**Recommendation:** Do not distill the raw "correct" set. If used at all, filter to bit-EXACT records
(re-grade with string equality, not the decimal-tolerance grader), then further filter to traces that
explicitly verify a derived rule against all examples and apply it to the query. Expect to retain only a
few hundred records at best, and a few dozen high-quality ones. gptoss should be dropped entirely.

---

## 1. Grader false positives — the central problem

The local grader (`eval/grader.py`, mirroring the official competition metric) extracts the boxed answer and
accepts it if it matches ground truth as a string **or within a relative numerical tolerance** (local `REL_TOL=1e-3`;
official competition tolerance is `1e-2` — even looser). The capture's `correct` flag is exactly
`is_correct(boxed(extracted), answer)` — I reconstructed it with **0 disagreements** across all 16,773 records.

**The fatal interaction:** an 8-bit answer like `00111110` is parsed by the numeric fallback as the *decimal
integer* `111110`. Ground truth `00111111` parses to `111111`. Relative difference = `1/111111 ≈ 9e-6`, far inside
both `1e-3` and `1e-2`. So **bit-flipped near-misses are graded "correct".** Because these decimal values are
~6–8 digits, the tolerance window (±0.1–1%) spans many low-order bit flips.

Verified directly:
```
is_correct("\boxed{00111110}", "00111111") -> True   # 1 bit off
is_correct("\boxed{11001100}", "11011111") -> True   # 3 bits off; reldiff 0.00091
```

Re-classifying every flagged-correct record by bit-exactness:

| model | flagged correct | bit-EXACT | tolerance-only (wrong bits) | FP Hamming dist (1/2/3/4/5) |
|---|---|---|---|---|
| deepseek | 706 | 258 | 448 | 188/119/116/24/1 |
| glm | 453 | 166 | 287 | 106/78/72/17/2 |
| gptoss | 631 | 237 | 394 | 156/78/131/28/1 |
| mistral | 609 | 235 | 374 | 124/106/119/19/4 |
| qwen | 132 | 39 | 93 | 32/9/12/2/0 |
| **TOTAL** | **2,531** | **935 (36.9%)** | **1,596 (63.1%)** | — |

All 1,596 tolerance-only records fall within even the stricter `1e-3`; none exceed `1e-2`. So they are
"correct" under the real competition too — but they are the **wrong 8-bit answer**. For a metric, that's points;
for **distillation, it teaches the model to emit near-miss bit strings**, which is exactly the wrong lesson.

## 2. Reasoning validity — SOUND / SHAKY / LUCKY

Reading traces shows that **bit-exactness does not imply sound reasoning.** Signals across traces with reasoning text:
only **5–11% of traces ever claim a rule that "matches all examples"**; the large majority end in explicit
guessing ("I'll guess from symmetry", "plausible bit-mixing function", "most structurally similar example").

Manual + semi-automated classification (verify-all-examples language AND no terminal capitulation AND bit-exact)
yields a candidate SOUND set of only **~21–28 unique (model,problem) traces**, and hand-reading shows even some of
those are rationalizations. Realistic genuinely-SOUND yield: **~15–25 distinct problems (~0.5–1% of flagged-correct).**

**Representative SOUND trace (deepseek `3456da40`, bit-exact, GT `00001111`)** — derives, verifies, applies:
> "Ex5: Input `00101100` >> 2 = `00001011` matches output `00001011`. … Ex7: `00011100` >> 2 = `00000111` matches.
> **Yes! All match perfectly.** The transformation is simply a **right shift by 2 bits**. … `00111101` >> 2 = `0x0F` =
> `00001111`. \boxed{00001111}"

My brute-force verifier independently confirms `SHR2` is the unique consistent rule and predicts `00001111` = GT.

**Representative SOUND trace (deepseek `214d0570`, bit-exact, GT `10111111`)** — discovers an MSB-conditional rule:
> "Example 2 (`10101011`) starts `10` → `10111111` ✓. Example 5 (`00000110`) starts `00` → `01111111` ✓.
> All others … → `11111111` ✓. Yes, that fits perfectly. Apply to `10010111`: starts `10` → `10111111`. \boxed{10111111}"

**Representative LUCKY-GUESS that was graded correct (deepseek `0ec17d2e`, bit-exact GT `00111111`):**
> "Given the time, I'll pick a plausible symmetric output … `10011111 → 00111111`. … from symmetry. \boxed{00111111}"
Bit-exact by luck — no rule derived. Distilling this teaches the model to capitulate-then-guess.

**Representative SHAKY rationalization (glm `2cf45d07`, bit-exact GT `10101010`):**
> "Combined with the parity constraint and the LSB rule, `10101010` is the most consistent prediction … matches the
> output of the most structurally similar example (Example 6). \boxed{10101010}"
No clean rule (my verifier finds 0 grammar rules); patched together from parity/"similar example" heuristics.

**Hallucinated rationalization (gptoss `108e69ef`, 3 bits off, accepted by tolerance):**
> "The given pairs correspond exactly to the **AES S-box** mapping … 0x74 → 0xC5, i.e. 11000101." 
False (these are random per-bit rules, not AES); produced a wrong answer the tolerance accepted. Toxic to distill.

## 3. Distillability — token budget & degeneracy

| model | median tok | p90 | over 7680 | stub traces (<300 chars, no reasoning) | truncated (max_tokens) |
|---|---|---|---|---|---|
| deepseek | 3,480 | 4,497 | 0 | 0 | 0 |
| glm | 1,456 | 5,711 | 17 (3.8%) | 0 | 11 |
| gptoss | 1,230* | 1,839* | 0 | **619 (98%)** | 0 |
| mistral | 4,960 | 6,678 | 12 (2.0%) | 0 | 3 |
| qwen | 8,192 | 8,192 | **74 (57%)** | 0 | 67 |

- **gptoss is unusable:** `out_tokens`/`reasoning_chars` are large (2k–8k), but the saved `answer_text` is just
  `\boxed{XXXXXXXX}` (median 16 chars). The chain-of-thought went to a reasoning channel that **was not persisted**;
  only the final boxed string remains. 619/631 "correct" gptoss traces carry **no derivation to distill**.
- **qwen** mostly hit the 8192 max-token cap (57% over budget, 67 truncated) — looping/over-long; low usable yield.
- deepseek/mistral fit the 7,680-token budget well; glm mostly fits.
- **No `\boxed{your answer}`-style placeholders** were found in any model (good — the grader was not fooled by literal placeholders).
- No catastrophic verbatim looping detected outside the qwen truncations.

## 4. Cross-model consistency (luck vs real rule)

- 132 distinct problems have ≥1 flagged-correct trace; only **69 (52%)** have any **bit-exact** correct answer —
  the other 63 were "solved" purely by tolerance near-misses.
- Only **39 problems** are bit-exact-solved by ≥2 models.
- **108/132 problems (82%)** have **disagreeing** extracted answers among their flagged-correct records
  (median **3 distinct** bit strings per problem). Models do **not** converge on one rule → the "correct" flags are
  dominated by independent decimal-tolerance luck, not a shared recovered transform.

## 5. Ground-truth cross-check (dataset is sound)

I wrote a brute-force verifier (`puzzle_team/results/bitman_verify.py`) searching a constrained grammar
(identity / NOT / reverse / ROL/ROR/SHL/SHR k=1..7 / cyclic neighbor XOR / majority, combined via AND/OR/XOR ± NOT)
for a rule consistent with **all** example pairs, then predicting the query.

On a 200-problem sample of `puzzle_team/data/bit_manipulation.jsonl`: a consistent rule was found within the
(deliberately limited) grammar for **127/200** problems; for **126/127** the dataset's ground-truth answer **equals
the rule's prediction** (the lone miss differs by 1 bit and uses a rule outside my grammar). The 73 "no rule found"
are simply rules richer than my grammar, not inconsistencies. **Conclusion: the dataset ground truth is consistent
with its own examples — the data is sound; the contamination is in the captures + tolerant metric.**

---

## Filtering criteria, if this corpus is salvaged

1. **Re-grade with string equality only** (drop the decimal-tolerance path): keep the 935 bit-exact records.
2. **Drop gptoss entirely** (no reasoning text saved).
3. **Drop truncated / over-budget** traces (qwen max_token hits; anything > 7,680 tok).
4. **Keep only traces that derive a rule and verify it on all examples**, and exclude terminal-guess language
   ("I'll guess", "from symmetry", "plausible", "most similar example"). This is the expensive but essential step;
   expect to retain only tens of high-quality traces (deepseek + mistral are the best sources).
5. Optionally re-derive each kept trace's rule with a verifier to confirm the boxed answer is bit-exact AND
   rule-justified before distilling.

**Net:** as-is the corpus would teach near-miss guessing and (via gptoss) answer-only shortcutting. After strict
filtering it yields a small but clean SOUND seed set; consider regenerating captures with (a) a string-exact grader
and (b) full reasoning-text persistence before scaling distillation.
