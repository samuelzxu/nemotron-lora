"""bit_manipulation solver — per-output-bit boolean inference.

Model (matches the winner's generator family): there is ONE global combine family
F in {I, NOT, 0, 1, AND, OR, XOR, AND-NOT, OR-NOT, XOR-NOT}. Each output bit j is a
function of two input bit positions (p_j, s_j):
    out[j] = F( in[p_j], NOT? in[s_j] )
For unary/constant families the source bits are irrelevant.

We infer, per output bit independently, which (family, p, s) reproduce that bit across
ALL examples, then combine. If multiple families fit a column we keep them and resolve by
consensus on the query. This subsumes whole-word shifts/rotations (a shift is just a fixed
source-bit mapping) and the SHL/SHR + combine tail.
"""
import re
from collections import Counter

N = 8


def parse(prompt):
    body = prompt.split("input -> output:")[-1].split("Now,")[0]
    exs = []
    for line in body.strip().split("\n"):
        if "->" in line:
            a, b = line.split("->")
            a, b = a.strip(), b.strip()
            if re.fullmatch(r"[01]{8}", a) and re.fullmatch(r"[01]{8}", b):
                exs.append((a, b))
    m = re.search(r"determine the output for:\s*([01]{8})", prompt)
    q = m.group(1) if m else None
    return exs, q


def _bin(a, b, fam):
    if fam in ("AND", "AND-NOT"):
        return a & b
    if fam in ("OR", "OR-NOT"):
        return a | b
    if fam in ("XOR", "XOR-NOT"):
        return a ^ b
    raise ValueError(fam)


PAIR = ("AND", "OR", "XOR", "AND-NOT", "OR-NOT", "XOR-NOT")


def _col_candidates(inputs, target_col):
    """Return list of callables (taking an input int-bit-list) producing this output column bit."""
    cands = []
    n = len(inputs)
    # unary / constant first (cheapest, most general)
    # identity from some source position
    for p in range(N):
        if all(inputs[k][p] == target_col[k] for k in range(n)):
            cands.append(("I", p, None))
    for p in range(N):
        if all((1 - inputs[k][p]) == target_col[k] for k in range(n)):
            cands.append(("NOT", p, None))
    if all(c == 0 for c in target_col):
        cands.append(("0", None, None))
    if all(c == 1 for c in target_col):
        cands.append(("1", None, None))
    # pairwise families
    for fam in PAIR:
        inv = fam.endswith("-NOT")
        for p in range(N):
            for s in range(N):
                ok = True
                for k in range(n):
                    a = inputs[k][p]
                    b = inputs[k][s]
                    if inv:
                        b = 1 - b
                    if _bin(a, b, fam) != target_col[k]:
                        ok = False
                        break
                if ok:
                    cands.append((fam, p, s))
    return cands


def _apply(fam, p, s, bits):
    if fam == "I":
        return bits[p]
    if fam == "NOT":
        return 1 - bits[p]
    if fam == "0":
        return 0
    if fam == "1":
        return 1
    a = bits[p]
    b = bits[s]
    if fam.endswith("-NOT"):
        b = 1 - b
    return _bin(a, b, fam)


def solve(prompt):
    exs, q = parse(prompt)
    if not exs or q is None:
        return None
    inputs = [[int(c) for c in a] for a, _ in exs]
    outcols = [[int(b[j]) for _, b in exs] for j in range(N)]
    qbits = [int(c) for c in q]

    # First try: a SINGLE global family that fits every column (the winner's constraint).
    # Collect per-column candidate families.
    per_col = [_col_candidates(inputs, outcols[j]) for j in range(N)]
    if any(len(c) == 0 for c in per_col):
        return None

    # Prefer a globally-consistent family. Find families present in all columns.
    fam_sets = [set(c[0] for c in col) for col in per_col]
    common = set.intersection(*fam_sets) if fam_sets else set()
    # Order preference: pairwise sym, then asym, then unary/const (more constrained = more reliable)
    pref = ["XOR", "AND", "OR", "XOR-NOT", "AND-NOT", "OR-NOT", "NOT", "I", "0", "1"]

    def build_with_family(fam):
        out = []
        for j in range(N):
            choices = [c for c in per_col[j] if c[0] == fam]
            if not choices:
                return None
            # consensus across all candidate source-pairs for this column on the query bit
            votes = Counter(_apply(*c, qbits) for c in choices)
            out.append(str(votes.most_common(1)[0][0]))
        return "".join(out)

    for fam in pref:
        if fam in common:
            r = build_with_family(fam)
            if r:
                return r

    # Fallback: per-column independent consensus over ALL candidate rules.
    out = []
    for j in range(N):
        votes = Counter(_apply(*c, qbits) for c in per_col[j])
        out.append(str(votes.most_common(1)[0][0]))
    return "".join(out)
