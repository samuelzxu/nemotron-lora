# Leaderboard — best solver + rate per HARD category

Winner baselines (to beat): cryptarithm_deduce 8.2% | cryptarithm_guess 6.7% | equation_numeric_guess 13.2% | bit_manipulation 85.1% (117 unsolved tail) | equation_numeric_deduce 90.6% (34 unsolved).

| Category | Best solver | Overall rate | rule_unknown rate | Approach |
|---|---|---|---|---|
| cryptarithm_deduce | cryptarithm_csp.py | ~2% (oracle cap) | ~1.7% (oracle: 2/120) | intractable: only 2/120 rule_unknown admit ANY symbol->digit arithmetic map |
| cryptarithm_guess | — | intractable | 0% | query op-char novel + map randomized per-problem => underdetermined |
| equation_numeric_guess | — | intractable (=winner 13.2%) | 0% | best fixed guess = "always absdiff" = 13.2% = winner's rate; no example info |
| bit_manipulation | bitman_perbit.py | 60% (perbit only) | 9.4% (11/117) | per-output-bit boolean (family + 2 source bits + NOT), global-family consensus |
| equation_numeric_deduce | equation_numeric_ops.py | 36.7% | 32.4% (11/34) | op-char->op family x {operand-rev, result-rev}, fit same-char examples |

NOTE: bitman/eq_deduce overall rates are BELOW the winner because these solvers target the
rule_unknown tail, not reproducing the winner's solved ground. The rule_unknown column is the
"new ground" metric that matters for the roof.

Harness: `foundation/.venv/bin/python puzzle_team/harness.py <category> puzzle_team/solvers/<name>.py [--status rule_unknown]`
