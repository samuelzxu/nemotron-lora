"""equation_numeric solver (deduce + guess).

Model: LHS is `AB op CD` with literal digits. The op char maps to ONE numeric operation
from a candidate family, with optional flags: reversed operands (a<->b) and/or reversed
result string. The output is the result string, possibly PREFIXED by the op char (observed
in the `guess` variant, e.g. answers like '*53', '17/').

We parse the examples, find every (op_name, rev_operands, rev_result, prefix_mode) tuple that
reproduces ALL examples sharing the query's op char, then apply to the query.
"""
import re


def parse(prompt):
    body = prompt.split("examples:")[-1].split("Now,")[0]
    exs = []
    for line in body.strip().split("\n"):
        if " = " in line:
            l, r = line.split(" = ")
            l, r = l.strip(), r.strip()
            m = re.fullmatch(r"(\d\d)(\D)(\d\d)", l)
            if m:
                exs.append((m.group(1), m.group(2), m.group(3), r))
    qm = re.search(r"determine the result for:\s*(\d\d)(\D)(\d\d)", prompt)
    if not qm:
        return exs, None
    return exs, (qm.group(1), qm.group(2), qm.group(3))


def candidates(sa, sb):
    a, b = int(sa), int(sb)
    out = []
    out.append(("concat", sa + sb))
    out.append(("revconcat", sb + sa))
    out.append(("add", str(a + b)))
    out.append(("absdiff", str(abs(a - b))))
    out.append(("negabsdiff", str(-abs(a - b))))
    out.append(("sub", str(a - b)))
    out.append(("revsub", str(b - a)))
    out.append(("mul", str(a * b)))
    out.append(("mul+1", str(a * b + 1)))
    out.append(("mul-1", str(a * b - 1)))
    out.append(("add+1", str(a + b + 1)))
    out.append(("add-1", str(a + b - 1)))
    out.append(("sub+1", str(a - b + 1)))
    out.append(("sub-1", str(a - b - 1)))
    if a and b:
        big, small = max(a, b), min(a, b)
        out.append(("maxmodmin", str(big % small)))
    if b:
        out.append(("div", str(a // b)))
        out.append(("mod", str(a % b)))
    if a:
        out.append(("revdiv", str(b // a)))
        out.append(("revmod", str(b % a)))
    d1, d2, d3, d4 = int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1])
    out.append(("digabsdiff", str(abs(d1 - d3)) + str(abs(d2 - d4))))
    out.append(("digaddmod10", str((d1 + d3) % 10) + str((d2 + d4) % 10)))
    out.append(("digsubmod10", str((d1 - d3) % 10) + str((d2 - d4) % 10)))
    out.append(("crossmul", str(d1 * d3 + d2 * d4)))
    out.append(("crossmulrev", str(d1 * d4 + d2 * d3)))
    out.append(("digmul", str(d1 * d3) + str(d2 * d4)))
    out.append(("digmulrev", str(d1 * d4) + str(d2 * d3)))
    out.append(("digsumdiff", str((d1 + d2) - (d3 + d4))))
    out.append(("digsumsum", str((d1 + d2) + (d3 + d4))))
    out.append(("digproddiff", str(d1 * d2 - d3 * d4)))
    out.append(("digprodsum", str(d1 * d2 + d3 * d4)))
    det = d1 * d4 - d2 * d3
    out.append(("det", str(det)))
    out.append(("absdet", str(abs(det))))
    return dict(out)


def _rev(s):
    if s.startswith("-"):
        return "-" + s[1:][::-1]
    return s[::-1]


def _produce(sa, op, sb, name, rev_ops, rev_res, prefix):
    if rev_ops:
        sa, sb = sb, sa
    cand = candidates(sa, sb)
    if name not in cand:
        return None
    res = cand[name]
    if rev_res:
        res = _rev(res)
    if prefix == "op":
        res = op + res
    return res


def solve(prompt):
    exs, q = parse(prompt)
    if not exs or q is None:
        return None
    qa, qop, qb = q
    # restrict to examples sharing the query op char if any exist; else use all
    same = [e for e in exs if e[1] == qop]
    train = same if same else exs

    names = list(candidates("00", "00").keys())
    best = None
    for name in names:
        for rev_ops in (False, True):
            for rev_res in (False, True):
                for prefix in (None, "op"):
                    ok = True
                    for (sa, op, sb, target) in train:
                        got = _produce(sa, op, sb, name, rev_ops, rev_res, prefix)
                        if got != target:
                            ok = False
                            break
                    if ok:
                        ans = _produce(qa, qop, qb, name, rev_ops, rev_res, prefix)
                        if ans is not None:
                            # prefer the first consistent rule; collect for consensus
                            if best is None:
                                best = ans
                            elif best != ans:
                                # ambiguity: keep the simpler (no prefix, no rev) earlier-found
                                pass
    return best
