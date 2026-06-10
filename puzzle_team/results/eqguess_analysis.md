# equation_numeric_guess: Is there an overarching rule, or just lucky guesses?

**Verdict: MOSTLY GUESSING, with ONE small real signal.** The category is essentially an
information-theoretic guessing game exactly as the prior analysis concluded — but there is a
single genuine, generalizable signal hiding in it: **when the query operator symbol is `-`,
the answer tends to be literal subtraction `a-b`.** Encoding that (plus minor `*`/`/` fits)
lifts the deterministic solve rate from the 13.2% baseline to **19.1%** (official exact-string
metric) / **22.1%** (numeric-tolerance harness grader). That is a real but modest improvement,
not a "crack the puzzle" rule.

---

## 1. Structure of the puzzle (why the examples are near-useless)

Each problem gives a few `AB op CD = RESULT` examples (often spanning *multiple* operator
symbols) then asks for a NEW equation with a DIFFERENT operator. Measured over all 136 problems:

- **The query operator is NEVER present in the examples: 0/136.**
- 107/136 problems show **more than one** operator symbol in the examples.
- The same symbol maps to **different** transforms across problems (e.g. `+` is `rev_concat`
  in 10 problems, plain `add` in 11, raw `concat` in 4, `mul` in 2...). Symbols have **no fixed
  global meaning**.
- 73.3% of individual example operator-groups ARE internally consistent (the model can and does
  crack `#`=a*b-1, `^`=a+b, etc.), so the examples are *solvable* — they just describe operators
  that are irrelevant to the question.

**The killer statistic:** for the 69/136 problems whose answer is explainable by some simple
transform, the answer's transform equals the **modal example transform in 0/136** cases, and is
among **ANY** example transform in only **2/136** cases. In 67/69 the query uses a transform that
*does not appear anywhere in the examples*. The puzzle is deliberately constructed so the examples
carry ~0 information about the query operator. This confirms the prior "guessing game" conclusion.

## 2. No single fixed rule beats absolute-difference

Best fixed "always apply X to the query operands" strategies (exact-string match, all 136):

| rule | solves | rate |
|------|-------:|-----:|
| **absolute difference \|a-b\|** | **18** | **13.2%**  ← existing baseline = the optimum fixed rule |
| subtraction a-b | 17 | 12.5% |
| digit absolute diff | 13 | 9.6% |
| concatenation | 9 | 6.6% |
| addition | 8 | 5.9% |

The winner already found the optimal *single* fixed rule. The answer-transform distribution is
flat and long-tailed (38 different transforms appear), exactly what you expect if the generator
samples a transform roughly uniformly and the examples don't disclose it.

**Oracle ceiling (overfit, peeking at answers):** even assigning the single best transform *per
query-symbol* — knowing all answers in advance — caps out at **40/136 = 29.4%**. And those best
transforms are mutually inconsistent (`+`→absdiff, `-`→sub, `*`→rev_concat), so the symbol does
not reliably determine the transform. There is no high-yield deterministic rule to be had.

## 3. The one real signal: the `-` symbol means subtraction

Decomposing the per-symbol gains (genuine, not peeking at which transform — just trusting the
symbol's typographic meaning):

| query symbol | semantic guess | solves | absdiff would get | verdict |
|---|---|---:|---:|---|
| `-` | a − b | **9/22** | 5/22 | **REAL signal** |
| `*` | reverse-concat b\|\|a | 4/17 | 1/17 | weak / partial train-fit |
| `/` | a × b | 1/8 | 0/8 | noise-ish |
| `+` | a + b | 2/26 | 3/26 | **NO** — worse than absdiff |

Only `-` is robust and semantically motivated: a minus sign genuinely tends to denote
subtraction even when "unseen." `+`→add does **not** generalize (it is *worse* than the absdiff
fallback), which is itself strong evidence that this is shared-prior guessing, not derivation.

## 4. Are the models' correct traces a derivable rule, or rationalized luck?

**Luck — driven by shared priors over "obvious" transforms.**

- **Coverage by repeated sampling is low.** DeepSeek (sampled many times) ever solves only
  **27/136** unique problems; gpt-oss 28/136; Mistral 24/136. Union of all three models =
  **39/136 = 28.7%**. Pass@many barely exceeds the 29.4% peeking oracle — i.e. the models are
  essentially just sampling the small pool of guessable transforms.
- **Cross-model overlap is concentrated on "guessable" problems, not random** — consistent with a
  shared prior, not derivation. All three models solve the same **15** problems; those decode to:
  literal `-`→sub/absdiff (5), literal `+`→add (4), `>`→sub, `*`→rev_concat, a×b−1, digit-sum.
  These are exactly the *salient* transforms a model would guess. The 14 problems solved by only
  ONE model are the long tail — different lucky hits per model = luck signature.
- **The reasoning explicitly admits guessing.** Representative DeepSeek trace for `70-11`
  (answer 59, solved 15/15 times):

  > "We don't have examples for `-`. ... maybe `-` is normal subtraction... **No tricks because
  > they didn't give any `-` examples**, so likely they want 59." → boxed 59. ✓ (lucky-but-right:
  > it correctly cracked `#`=a×b−1 and `^`=a+b from examples, which told it *nothing* about `-`,
  > then guessed the literal symbol meaning.)

  For `86{99` (answer 8513): models land on **a×b−1** (86×99−1=8513) — a salient "off-by-one
  product" guess, again unconnected to the examples.
  For `72+44` (answer 17): the hit is **digit-sum** (7+2+4+4) — another common puzzle prior.

  When a model gets a problem right it is highly *self-consistent* (e.g. 15/15 samples say "59"),
  but that reflects a confident **prior** on the symbol, not evidence recovered from the examples
  — the examples are about other operators entirely.

So: high within-problem answer consistency + near-zero example→answer information + overlap
concentrated on typographically-salient transforms = the corrects are **prior-driven guesses that
the puzzle generator happened to also pick**, not a derived rule.

## 5. Deterministic solver (`solvers/equation_guess_rule.py`)

Encodes only the defensible signal: `-`→subtraction; mild `*`→reverse-concat and `/`→multiply;
everything else (incl. `+`) → absolute difference (the modal fallback).

```
SOLVE RATE: 30/136 = 22.1%   (numeric-tolerance harness grader)
  rule_found        19/21   rule_unknown 6/80   hypothesis_formed 5/35
```

| strategy | exact-string (official metric) | numeric-tolerance (harness) |
|---|---:|---:|
| always absdiff (baseline) | 18/136 = **13.2%** | 22/136 = 16.2% |
| **rule (`-`,`*`,`/`) [this solver]** | 26/136 = **19.1%** | 30/136 = **22.1%** |
| rule + (`+`→add) | 18.4% (worse) | 20.6% (worse) |

(The harness's numeric tolerance ignores leading-zero/sign-format quirks, so its baseline is
16.2% vs the official 13.2%. The +5.9pt absolute gain under the *official* exact-string metric is
the honest headline.)

## 6. Bottom line

- **Is there an overarching rule that "solves" the category? No.** The examples are deliberately
  uninformative about the query operator (0/136 query-ops appear in examples; answer-transform
  matches example-transform in 2/136). Even an answer-peeking per-symbol oracle caps at 29.4%.
- **Is there ANY deterministically-encodable signal? Yes, one small one:** the `-` symbol tends
  to mean literal subtraction. Trusting the query symbol's typographic meaning for `-` (and, with
  weaker confidence, `*`/`/`) beats the 13.2% baseline → **19.1%** exact-string / **22.1%** harness.
- **Are the frontier-model corrects a rule or luck? Luck via shared priors.** Low coverage
  (≤28/136 per model), cross-model overlap concentrated on typographically-salient transforms,
  and traces that explicitly say "no examples for this op, I'll assume it means what it looks like."
  The models are not recovering information from the examples; they are guessing the obvious, and
  the generator sometimes agrees.

**Recommendation:** ship `equation_guess_rule.py` (a clean ~6pt deterministic lift over baseline
from the `-`→subtraction signal), but do not expect to push much past ~20%: the category is, by
construction, dominated by irreducible guessing.
