"""Answer extraction + grading mirroring the competition metric.

Judge semantics (from competition_description.txt):
- Extract the final answer, prioritizing content inside \\boxed{...}.
- Fall back to other heuristic patterns, then the last numeric value found.
- Correct if it matches ground truth either exactly as a string OR within a
  relative numerical tolerance.

NOTE: reconcile REL_TOL and the fallback order with the OFFICIAL metric once the
nemotron-packages 'metric' module is unpacked locally (US-005). This is a faithful
reimplementation of the documented semantics, used for the local held-out harness.
"""
from __future__ import annotations
import re
from typing import Optional

REL_TOL = 1e-3  # relative numeric tolerance; reconcile with official metric

_BOXED_RE = re.compile(r"\\boxed\s*\{")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _extract_boxed(text: str) -> Optional[str]:
    """Return the content of the LAST \\boxed{...}, handling nested braces."""
    last = None
    for m in _BOXED_RE.finditer(text):
        i = m.end()  # position just after the '{'
        depth = 1
        buf = []
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(c)
            i += 1
        if depth == 0:
            last = "".join(buf).strip()
    return last


def extract_answer(text: str) -> Optional[str]:
    """Boxed first, then last numeric value as the documented fallback."""
    if text is None:
        return None
    boxed = _extract_boxed(text)
    if boxed is not None and boxed != "":
        return boxed
    nums = _NUM_RE.findall(text)
    if nums:
        return nums[-1]
    return None


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s.strip().replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s).strip())


def is_correct(prediction_text: str, ground_truth: str, rel_tol: float = 1e-2) -> bool:
    """Faithful replica of the official NVIDIA metric (winner_repo/reasoning.py compare_answer).

    CRITICAL: binary-string answers ([01]+) are compared STRICTLY as strings — NO numeric
    tolerance. (Earlier this fell through to the numeric path and graded bit-flipped near-misses
    as correct.) Non-binary numeric answers use rel_tol=1e-2; everything else is case-insensitive
    string equality.
    """
    pred = extract_answer(prediction_text)
    if pred is None:
        return False
    stored = str(ground_truth).strip()
    pred = str(pred).strip()
    # binary-string answers: strict string comparison (matches the official metric)
    if re.fullmatch(r"[01]+", stored):
        return pred.lower() == stored.lower()
    try:
        import math
        return math.isclose(float(stored), float(pred), rel_tol=rel_tol, abs_tol=1e-5)
    except Exception:
        return pred.lower() == stored.lower()


if __name__ == "__main__":
    # self-tests (no model / data needed)
    cases = [
        ("reasoning... \\boxed{00000100}", "00000100", True),
        ("first \\boxed{12} then \\boxed{42}", "42", True),       # last boxed wins
        ("the answer is \\boxed{ XIV }", "XIV", True),            # whitespace-insensitive
        ("\\boxed{3.14159}", "3.1416", True),                     # within rel tol
        ("\\boxed{3.20}", "3.14", False),                         # outside rel tol
        ("no box, value 99 at end", "99", True),                  # numeric fallback
        ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}", True),          # nested braces
        ("\\boxed{cat}", "dog", False),
    ]
    ok = 0
    for text, gt, want in cases:
        got = is_correct(text, gt)
        status = "ok" if got == want else "FAIL"
        if got == want:
            ok += 1
        print(f"[{status}] extract={extract_answer(text)!r} gt={gt!r} -> {got} (want {want})")
    print(f"\n{ok}/{len(cases)} passed")
