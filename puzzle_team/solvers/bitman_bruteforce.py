"""bit_manipulation solver: brute-force search over compositions of bitwise transforms.

Receives ONLY the prompt. Parses 8-bit input->output examples, searches for a composition
(depth up to 3) of primitive transforms that maps every example input to its output, then
applies it to the query. No token limit -> wide search is fine.
"""
import re
from itertools import product

N = 8
MASK = (1 << N) - 1


def rotl(x, k):
    k %= N
    return ((x << k) | (x >> (N - k))) & MASK if k else x


def rotr(x, k):
    k %= N
    return ((x >> k) | (x << (N - k))) & MASK if k else x


def shl(x, k):
    return (x << k) & MASK


def shr(x, k):
    return (x >> k) & MASK


def _prims():
    p = [("id", lambda x: x)]
    for k in range(1, N):
        p.append((f"rotl{k}", (lambda k: lambda x: rotl(x, k))(k)))
        p.append((f"rotr{k}", (lambda k: lambda x: rotr(x, k))(k)))
        p.append((f"shl{k}", (lambda k: lambda x: shl(x, k))(k)))
        p.append((f"shr{k}", (lambda k: lambda x: shr(x, k))(k)))
    p.append(("not", lambda x: (~x) & MASK))
    return p


PRIMS = _prims()


def parse(prompt):
    body = prompt.split("input -> output:")[-1]
    body = body.split("Now,")[0]
    exs = []
    for line in body.strip().split("\n"):
        if "->" in line:
            a, b = line.split("->")
            a, b = a.strip(), b.strip()
            if re.fullmatch(r"[01]{8}", a) and re.fullmatch(r"[01]{8}", b):
                exs.append((int(a, 2), int(b, 2)))
    m = re.search(r"determine the output for:\s*([01]{8})", prompt)
    q = int(m.group(1), 2) if m else None
    return exs, q


COMBINERS = [
    ("xor", lambda a, b: a ^ b),
    ("or", lambda a, b: a | b),
    ("and", lambda a, b: a & b),
]


def _candidates_depth(inputs, outputs):
    """Yield callables f such that f(inp)==out for all examples.

    Family 1: single primitive p (depth-1).
    Family 2: combiner(p1(x), p2(x))  (covers e.g. x ^ shr(x), majority-ish via and/or/xor).
    Family 3: combiner(p1(x), combiner2(p2(x), p3(x)))  (3-transform tail).
    Also chained primitives p2(p1(x)) up to depth 3.
    """
    pr = PRIMS
    # Family 1: single primitive
    for n1, f1 in pr:
        if all(f1(i) == o for i, o in zip(inputs, outputs)):
            yield (n1,), f1
    # chained primitives depth 2 and 3
    for n1, f1 in pr:
        v1 = [f1(i) for i in inputs]
        for n2, f2 in pr:
            v2 = [f2(v) for v in v1]
            if all(v == o for v, o in zip(v2, outputs)):
                yield (n1, n2), (lambda f1, f2: lambda x: f2(f1(x)))(f1, f2)
    # Family 2: combiner(p1(x), p2(x))
    for cn, cf in COMBINERS:
        for i1 in range(len(pr)):
            n1, f1 = pr[i1]
            v1 = [f1(x) for x in inputs]
            for i2 in range(i1, len(pr)):
                n2, f2 = pr[i2]
                v2 = [f2(x) for x in inputs]
                if all(cf(a, b) == o for a, b, o in zip(v1, v2, outputs)):
                    yield (cn, n1, n2), (lambda cf, f1, f2: lambda x: cf(f1(x), f2(x)))(cf, f1, f2)


def search(exs):
    inputs = [i for i, _ in exs]
    outputs = [o for _, o in exs]
    found = []
    for desc, f in _candidates_depth(inputs, outputs):
        found.append((desc, f))
    return found


def _candidates_depth3(inputs, outputs):
    """Heavier search: combiner(p1(x), combiner2(p2(x), p3(x)))."""
    pr = PRIMS
    pv = [[f(x) for x in inputs] for _, f in pr]
    for cn, cf in COMBINERS:
        for c2n, c2f in COMBINERS:
            for i1 in range(len(pr)):
                v1 = pv[i1]
                for i2 in range(len(pr)):
                    v2 = pv[i2]
                    inner = [c2f(a, b) for a, b in zip(v2, [pv[i2][k] for k in range(len(inputs))])]
                    for i3 in range(i2, len(pr)):
                        v3 = pv[i3]
                        ok = True
                        for k in range(len(inputs)):
                            if cf(v1[k], c2f(v2[k], v3[k])) != outputs[k]:
                                ok = False
                                break
                        if ok:
                            yield ((cn, c2n, pr[i1][0], pr[i2][0], pr[i3][0]),
                                   (lambda cf, c2f, f1, f2, f3: lambda x: cf(f1(x), c2f(f2(x), f3(x))))(
                                       cf, c2f, pr[i1][1], pr[i2][1], pr[i3][1]))
                            return


def solve(prompt):
    exs, q = parse(prompt)
    if not exs or q is None:
        return None
    cands = search(exs)
    if cands:
        # consensus on query output
        from collections import Counter
        votes = Counter(f(q) for _, f in cands)
        best, _ = votes.most_common(1)[0]
        return format(best, "08b")
    # fall to depth-3 nested combiner
    inputs = [i for i, _ in exs]
    outputs = [o for _, o in exs]
    for desc, f in _candidates_depth3(inputs, outputs):
        return format(f(q), "08b")
    return None
