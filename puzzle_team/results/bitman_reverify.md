# Bit-Manipulation Capture — Independent Re-Verification

**Mandate:** Independently re-derive the key facts and decide whether to CONFIRM or OVERTURN the
v2 verdict ("do NOT distill the Bedrock bit_manipulation captures"). All numbers below were
computed from scratch with the **current** grader (`eval/grader.py`: binary `[01]+` answers
compared **string-exact, no numeric tolerance**), `foundation/.venv/bin/python`. Stored `correct`
flags were never trusted. Prior reports (`bitman_capture_audit.md` v1, `bitman_capture_quality_v2.md`
v2) were read only **after** forming my own numbers.

## VERDICT: **CONFIRM v2 — do NOT distill the Bedrock bit_manipulation captures.**

The captures are ~63% inflated, soundness is dominated by terminal capitulation-guessing (glm the
only partial exception), and — critically — they add **no coverage value** over the existing
1754-row dataset, which is 99.89% bit-exact and uniformly sound. This is the OPPOSITE of the
eq_guess situation: there is no pool of hard, sound, net-new-value traces here.

---

## Check 1 — Existing-data validity

| metric | value |
|---|---|
| bit_manip rows (`type=="bit_manipulation"`) | 1754 |
| bit-exact `generated_cot` under current grader | **1752 = 99.89%** |
| empty CoT | 0 |
| `00000000` share of training answers | 116 = 6.6% |

**CONFIRMS v2's 99.9% claim** (1752/1754). The training CoT is a clean, systematic per-bit rule
search (lay out columns → test operators on all examples → "Perfect match" → apply). Verified by
reading; zero guess-language in samples.

## Check 2 — Capture inflation (stored-correct vs re-graded bit-exact)

| model | stored correct | bit-EXACT (re-grade) | inflation | bit-exact rate |
|---|---|---|---|---|
| deepseek | 706 | **258** | 63.5% | 36.5% |
| glm | 453 | **166** | 63.4% | 36.6% |
| gptoss | 631 | **237** | 62.4% | 37.6% |
| mistral | 609 | **235** | 61.4% | 38.6% |
| qwen | 213 | **67** | 68.5% | 31.5% |

**CONFIRMS v2's ~63% inflation / ~37% real.** The stored `correct` field is contaminated by the old
decimal-tolerance grader accepting bit-flipped near-misses (e.g. `00111110` vs `00111111` parse to
decimals 111110 vs 111111, reldiff ~9e-6, inside tolerance).

### ⚠ Discrepancy with v1/v2 on qwen — MY NUMBERS SUPERSEDE THEIRS
Both prior reports call qwen "partial" (2525 rows / 132–138 stored-correct / 39–40 bit-exact). The
qwen file is now **complete**: 3600 rows = 150 problems × 24 gens, **213 stored-correct, 67
bit-exact**. The file was evidently finished after v1/v2 ran. My qwen figures are the correct
current ones. (qwen's truncation problem stands: 3318/3600 gens = 92% hit `max_tokens`.)

## Check 3 — Degenerate (`00000000`) share of bit-exact wins

| model | bit-exact | all-zeros (`00000000`) | non-degenerate | degen share |
|---|---|---|---|---|
| deepseek | 258 | 114 | 144 | 44% |
| glm | 166 | 74 | 92 | 45% |
| gptoss | 237 | 98 | 139 | 41% |
| mistral | 235 | 95 | 140 | 40% |
| qwen | 67 | 34 | 33 | 51% |

**CONFIRMS v2's all-zeros dominance flag.** ~40–51% of every model's bit-exact "wins" are the
trivial constant `00000000`, reachable by almost any "output 0 unless special case" rationalization.

## Check 4 — THE LOAD-BEARING COVERAGE CHECK

### 4a. The 150 captured problems are the HARD tail
Status mix of `bedrock/bitman_sample.jsonl`: **117 `rule_unknown`** (winner could NOT solve) +
**33 `hypothesis_formed`**. So yes — the captures target predominantly the hard tail.

### 4b. Distinct problems solved bit-exact, and net-new intersection

| model | distinct bit-exact | distinct non-degen bit-exact |
|---|---|---|
| deepseek | 37 | 31 |
| glm | 35 | 28 |
| gptoss | 34 | 27 |
| mistral | 40 | 33 |
| qwen | 11 | 8 |
| **POOLED (union)** | **69 / 150** | **62 / 150** |

Of the 62 pooled non-degenerate bit-exact problems, **47 are `rule_unknown`** (hard) and 15 are
`hypothesis_formed`.

**Intersection with the existing training set:**
- Match by problem ID: **0 overlap** (capture IDs and training IDs are disjoint namespaces).
- Match by exact prompt string: **0 overlap**.
- Match by whitespace-normalized prompt: **0 overlap**.
- Match by the sorted set of example-pairs (the rule fingerprint): **0 overlap**.

The training set and the capture set are **entirely disjoint problem instances** — randomly
generated puzzles with no shared rule/examples/query. So in the literal "net-new instance" sense,
**all 62 non-degenerate bit-exact capture problems are net-new instances**.

### Why this is NOT a distill signal (unlike eq_guess)
The eq_guess audit flipped because the captures contained *sound* solutions to *hard problems whose
training coverage was wrong/absent*. That logic does not transfer here, for two independent reasons:

1. **No coverage gap to fill.** Coverage here is not about specific instances (every instance is a
   fresh random draw) but about whether the model has clean worked examples of the *per-bit
   derivation skill*. The existing 1754 rows already provide that at **99.89% bit-exact** and
   uniformly sound. Adding 62 more disjoint instances of the same skill family adds quantity, not a
   missing capability. There is no "hard sub-family the training data gets wrong" — the training
   data is right.
2. **The capture "solutions" to the hard problems are mostly unsound (see Check 5).** A net-new
   *instance* only helps if it carries a *sound* derivation. For the hard (`rule_unknown`) problems,
   the captures overwhelmingly reach the right string by capitulation-guessing, not by a verified
   rule — so they would teach the wrong behavior even where they're "net-new."

## Check 5 — Soundness spot-check (read, not heuristic)

Sampled ~20 non-degenerate bit-exact traces across all 5 models. Guess-language prevalence in the
full non-degenerate bit-exact pools (lower bound on unsound, terms like
guess/plausible/stuck/running-out/most-likely/maybe):

| model | non-degen bit-exact traces | with guess-language | distinct problems |
|---|---|---|---|
| deepseek | 144 | 142 (99%) | 31 |
| glm | 92 | 50 (54%) | 28 |
| gptoss | 139 | 137 (99%) | 27 |
| mistral | 140 | 138 (99%) | 33 |
| qwen | 33 | 17 (52%) | 8 |

**glm is the only model with a meaningful sound rate (~half).** qwen's pool is an artifact — 24 of
33 traces are the single easy problem `3456da40` (logical SHR2, `hypothesis_formed`), so its low
guess-rate reflects one trivial problem, not breadth.

**Excerpts (terminal lines):**

- UNSOUND (deepseek `6cface63`, hard, GT `10010000`):
  > "Given the contradictions and time spent, I'll note that in many guess patterns, the outputs
  > often have bits 6 and 0 set ... but here for `00010010`, likely output is `10010000`."
  Bit-exact by luck; no verified rule.

- UNSOUND (gptoss `2cf45d07`, hard, GT `10101010`):
  > "Given difficulty, maybe answer is 01010101? Guess? ... Likely output maybe 10101010 (0xAA).
  > I'll answer that. \boxed{10101010}"
  Explicit guess.

- UNSOUND (mistral `288c7eca`, hard, GT `01111111`):
  > "Given the ambiguity, the most likely output for `10110101` is `10111111` or `01111111`. Given
  > that `10100110` → `01111111`, it's likely `01111111`."
  Analogy, not derivation.

- SOUND (glm `51007339`, hard, GT `10001110`): genuine per-bit derivation matching the training
  style —
  > "Bit 7: $y_7 = b_4$ ... Bit 6: Maj(b_5,b_7,b_3) ... Bit 5: $y_5=b_2$ ... Shift left 3 ... wrap ...
  > Result: 10001110. \boxed{10001110}"

- FALSE-POSITIVE-on-heuristic (glm `6cface63`, hard, GT `10010000`): contains verify markers but is
  actually a failed hypothesis the model abandons —
  > "AND: `01011100` AND `01000101` = `01000100`. Close to `01001000`, but bit 3 is 0 instead of 1
  > ..." — flails, no confirmed rule, lands on the answer anyway.

**Calibration:** an automated "has verify-marker AND no terminal-guess" heuristic flags 37 distinct
non-degenerate problems (31 hard) as having a "sound" trace — but manual reading shows it
**over-counts** (the glm `6cface63` case above). The honest realistic count of genuinely-sound,
non-degenerate, hard-problem traces is small and concentrated almost entirely in glm, roughly the
**~15–30 distinct problems** range v1/v2 estimated — and every one of them is the *same per-bit
derivation skill the training set already demonstrates 1752 times.*

## Reconciliation with v1 and v2

- **v2's three headline numbers (99.9% existing validity, ~63% inflation/~37% real, all-zeros
  dominance): CONFIRMED exactly.**
- **v2's coverage conclusion ("no meaningful coverage gap"): CONFIRMED**, and strengthened: I
  verified the ID/prompt/rule-fingerprint intersection is literally **0** — the sets are disjoint
  instances — so "redundant in capability" is the right frame, not "redundant in instance." The
  captures cannot fill a gap because the existing data has none.
- **v2's soundness read (capitulation-guessing dominant, glm best): CONFIRMED** by independent
  reading and guess-language prevalence.
- **qwen "partial / 132–138 correct / 39–40 bit-exact" (BOTH v1 and v2): OUTDATED.** The file is now
  complete (3600 rows, 213 stored-correct, 67 bit-exact). My numbers are correct; theirs predate the
  file's completion. This does not change the verdict (qwen is still 92% truncated and its
  non-degenerate sound yield is ~1 easy problem).
- **v1's gptoss "no reasoning text" claim** was already corrected by v2 (the `text` field does carry
  a CoT). I independently confirm gptoss `text` is non-null on its 631 stored-correct rows and
  contains real (but guess-heavy) scratch-work; I side with v2 on this.

## Conclusion

Distilling these captures would (1) require discarding ~63% of "correct" rows as bit-flipped
near-misses, (2) after that, fight a ~40–51% degenerate-all-zeros share, (3) then hand-filter
capitulation-guessing that dominates every model except glm, leaving only a few dozen genuinely
sound traces — all of which exercise the exact per-bit derivation skill the existing 99.89%-clean
1754-row dataset already covers densely. There is no hard sub-family the training data gets wrong
that the captures get right and sound. **CONFIRM v2: do not distill.**
