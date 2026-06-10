# Findings — decoded rules & what works/fails (cross-pollinate here)

## Structure (all cryptarithm + equation_numeric)
- Every LHS is `AB op CD` = 5 chars: 2-char left operand, 1 op char (pos 2), 2-char right operand.
- Output (RHS) length 1-4 chars.
- Prompt: lines `LHS = RHS`, then `Now, determine the result for: <query5>`.
- Parse: split on `examples:` then on `Now,`; each line splits on ` = `; query after `determine the result for:`.

## cryptarithm_deduce / guess
- Per-problem symbol alphabet. Hypothesis: symbol->digit 0-9 (bijective per problem), op char -> op.
- Winner ONLY did concat + (add/abs_diff/mul) with UNIQUE digit map -> 8.2%. Huge headroom.
- IDEAS (different workers, no dup):
  1. Full CSP/brute over symbol->digit bijection + op family {add, signed sub, abs_diff, mul, concat,
     rev_concat, div, mod, reverse}. Map consistent with ALL examples -> apply to query.
  2. Multiple op-chars per problem -> solve jointly; op semantics may vary by op char.
  3. Output may be a direct char-level transform of input symbols (not via digits). Check first.
  4. Negative / leading-zero / signed rendering.

## bit_manipulation (85% solved, 117 unsolved 3-transform tail)
- 8-bit->8-bit. Winner <=2-transform. Tail = 3 composed transforms incl SHL(x)/SHR(y), x+y<8.
- IDEA: brute search over composition depth<=3 of {ROTL k, ROTR k, SHL k, SHR k, NOT, XOR/AND/OR mask, id},
  verify all examples. No token limit.

## equation_numeric_deduce / guess
- LHS digits literal 0-9. Op char -> one of ~32 numeric ops w/ operand/result reversal variants.
- guess answers sometimes include op-char prefix (e.g. '*53','17/'). Investigate output formatting.

## Status log
(append worker findings below)

## FINAL VERDICT (manager, evidence-based)
- cryptarithm_deduce: ONLY 2/120 rule_unknown admit any symbol->digit map under standard ops
  {add,sub,absdiff,mul,div,mod,reverses,digitwise}. >10 symbols per problem in many cases makes a
  bijection impossible. The generator's rule for the tail is NOT standard digit-arithmetic.
  Roof on rule_unknown ~= 2%. INTRACTABLE with obvious models.
- cryptarithm_guess / equation_numeric_guess: query op-char is NOVEL (never in examples) and the
  op-char->operation map is randomized per-problem. Examples carry ZERO info about the query op.
  Proof: best fixed-guess for eq_numeric_guess = "always absdiff" = 13.2% = EXACTLY the winner's rate.
  These are information-theoretically a guessing game. Roof = winner's rate. INTRACTABLE.
- bit_manipulation: per-output-bit boolean model (global family + 2 source bits + optional NOT) gets
  9.4% (11/117) on the rule_unknown tail = NEW GROUND. Free per-column search reaches only 60% overall
  (winner 85% via tighter structured operand offsets that we did not fully reverse-engineer). The tail
  IS programmatic; with the winner's exact AP-offset structure one could likely push the tail higher.
- equation_numeric_deduce: op-family x {operand-reversal, result-reversal} solver gets 32.4% (11/34) on
  rule_unknown = NEW GROUND. Real headroom; richer op set would lift it further.

## Distillable-headroom ranking
1. equation_numeric_deduce rule_unknown (32%) -> worth generating CoT.
2. bit_manipulation rule_unknown (9%, likely more with structured search) -> worth it.
3. cryptarithm_* and equation_numeric_guess -> NOT worth deterministic generation (intractable).
