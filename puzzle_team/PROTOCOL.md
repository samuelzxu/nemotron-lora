# Puzzle-Cracking Team — Protocol

## Mission
Find the **roof**: the maximum per-category solve rate achievable on the HARD categories of the
NVIDIA Nemotron reasoning challenge, using **deterministic Python solvers + research** (Claude Code
agents, NOT Bedrock LLMs). We want to know how many of these puzzles are *programmatically solvable*
— that determines what's even worth encoding as training data.

## Ground truth & baselines (beat the winner)
`puzzle_team/data/<category>.jsonl` — each row: `{id, prompt, answer, winner_status, winner_submission}`.
The Open-Progress winner's solve rates are the baseline to beat:

| Category | N | Winner solve % | Unsolved by winner (rule_unknown) |
|---|---|---|---|
| cryptarithm_deduce | 659 | 8.2% | 559 |
| cryptarithm_guess | 164 | 6.7% | 128 |
| equation_numeric_guess | 136 | 13.2% | 80 |
| bit_manipulation | 1602 | 85.1% | 117 (the 3-transform tail) |
| equation_numeric_deduce | 596 | 90.6% | 34 |

`winner_status=rule_unknown` = problems the winner could NOT solve. **Every rule_unknown you crack is
new ground.** (cryptarithm is the biggest prize and the winner said it "requires guessing" — an open
research problem.)

## Solver contract
A solver is a Python module exposing `solve(prompt: str) -> str | None`. It receives **ONLY the
prompt** and must derive the answer from the prompt's examples alone.
- Save solvers to `puzzle_team/solvers/<descriptive_name>.py`.
- **Do NOT read `answer` or `winner_submission` inside a solver** — that's cheating; the harness uses
  them only for scoring. Solvers that peek will be rejected.
- Return `None` if you can't solve a given problem (better than a wrong guess for analysis, though the
  metric only rewards correct answers).

## Evaluation (the ONLY source of truth for solve rate)
```
foundation/.venv/bin/python puzzle_team/harness.py <category> puzzle_team/solvers/<name>.py
# options: --status rule_unknown   (score only the winner-unsolved subset)
#          --limit N --timeout SEC (per-problem seconds, default 5)
```
The harness prints `SOLVE RATE`, a breakdown by winner_status, and a `RESULT_JSON` line.

## Coordination
- **Leaderboard:** `puzzle_team/results/leaderboard.md` — the manager keeps the best solver + rate per
  category and a short note on the winning approach. Update it whenever a worker beats the current best.
- **Findings log:** `puzzle_team/results/findings.md` — append insights (decoded transformation
  families, what works/fails) so workers cross-pollinate instead of duplicating.
- Workers: announce your category+approach before starting (avoid two workers on the identical idea).

## Puzzle-type primers (start here, then research deeper)
- **cryptarithm_deduce / guess**: symbol-substitution + an arithmetic/transformation rule inferred from
  a few `LHS = RHS` examples over a symbol alphabet. Likely involves decoding a char->value map and an
  operator (concatenation, reverse, arithmetic). Research: cryptarithm / verbal-arithmetic solving,
  constraint propagation, the winner's `winner_repo/reasoners/cryptarithm.py` and
  `winner_repo/investigators/cryptarithm_deduce.py`.
- **bit_manipulation**: discover a per-output-bit boolean rule (ROT/SHL/SHR + AND/OR/XOR/NOT) from
  8-bit input->output examples. Winner solves ≤2-input-bit cases; the open tail is 3-transform with
  SHL(x)/SHR(y), x+y<8. Research: `winner_repo/reasoners/bit_manipulation.py`,
  `winner_repo/nemotron_context/winner_pub_p2.txt`. Brute-force search over expression templates is
  viable in Python (no token limit here!).
- **equation_numeric_deduce / guess**: infer the operator mapping two operands to a result from
  examples (32 operators, operand/result reversal variants). Winner's `reasoners/equation_numeric.py`.

## North star
Report, per category: best solve rate, rate on the `rule_unknown` subset, and the approach. The
aggregate "roof" tells us which hard categories have distillable headroom (solvable in code => we can
generate correct CoT) vs. are genuinely intractable.
