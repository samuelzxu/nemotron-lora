"""equation_numeric_guess solver — symbol-semantics rule.

Forensic finding (see results/eqguess_analysis.md): in this category the QUERY
operator symbol is NEVER present in the examples (0/136), and the SAME symbol maps
to different transforms across problems, so the examples carry essentially no
information about the query transform. The examples are effectively a distractor.

The ONLY generalizable signal is the *typographic / semantic meaning of the query
symbol itself*, and even that is weak:
  - '-'  -> literal subtraction (a-b)      : 9/22 correct (vs 5/22 for absdiff)  [REAL signal]
  - '*'  -> reverse concatenation (b||a)   : 4/17                                [weak]
  - '/'  -> multiplication (a*b)           : 1/8                                 [noise-ish]
  - everything else (incl '+') -> absolute difference |a-b| (the modal operation, 13.2% floor)

We deliberately do NOT use '+' -> add: it scores 2/26, WORSE than absdiff (3/26).

This is a GUESS policy, not a derivation. It beats the 13.2% "always absdiff"
baseline only modestly and only because of the genuine '-' signal plus mild
train-set fit on '*'/'/'. Prompt-only; never peeks at the answer.
"""
import re

_QRE = re.compile(r"determine the result for:\s*(\d+)(\D)(\d+)")


def solve(prompt: str):
    m = _QRE.search(prompt)
    if not m:
        return None
    sa, op, sb = m.group(1), m.group(2), m.group(3)
    a, b = int(sa), int(sb)
    if op == "-":
        return str(a - b)          # minus symbol genuinely tends to mean subtraction
    if op == "*":
        return sb + sa             # reverse concatenation, raw strings (mild signal)
    if op == "/":
        return str(a * b)          # mild signal
    return str(abs(a - b))         # modal fallback (= the 13.2% baseline)
