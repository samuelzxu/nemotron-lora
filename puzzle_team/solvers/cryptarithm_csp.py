"""cryptarithm_deduce/guess solver — constraint-propagation arithmetic.

Model: per-problem map symbol->digit (0-9), NOT necessarily injective. The op char maps to an
arithmetic op. left=AB, right=CD as 2-digit numbers; out is the result rendered back through the
same map (digit->symbol), possibly with a sign char. We CP-backtrack symbol assignments that satisfy
every example, then apply to the query. Emit only if the query answer is uniquely determined across
all consistent solutions (else None — better to abstain).

Pruning: assign symbols in order of first appearance; for each fully-instantiated example, verify the
arithmetic immediately. Cap solution enumeration. Designed to find the determinable subset only.
"""
import re

OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "absdiff": lambda a, b: abs(a - b),
    "mul": lambda a, b: a * b,
}
# concat handled separately (string-level), no arithmetic needed.


def parse(prompt):
    body = prompt.split("examples:")[-1].split("Now,")[0].strip()
    exs = []
    for line in body.split("\n"):
        if " = " in line:
            l, r = line.split(" = ")
            l, r = l.strip(), r.strip()
            if len(l) == 5:
                exs.append((l, r))
    m = prompt.split("determine the result for:")[-1].strip()
    q = m if len(m) == 5 else None
    return exs, q


def _render(val, inv):
    s = str(val) if val >= 0 else "-" + str(-val)
    out = []
    for ch in s:
        if ch == "-":
            out.append("-")
            continue
        d = int(ch)
        if d not in inv:
            return None
        out.append(inv[d])
    return "".join(out)


def _solve_op(exs, q, opname):
    """Try a single arithmetic op for ALL op chars. Return set of query answers."""
    op = OPS[opname]
    # symbol order by first appearance
    order = []
    seen = set()
    for l, r in exs:
        for ch in l[0] + l[1] + l[3] + l[4] + r:
            if ch not in seen:
                seen.add(ch)
                order.append(ch)
    for ch in q:
        if ch not in seen and ch != q[2]:
            seen.add(ch)
            order.append(ch)
    if len(order) > 11:
        return set()  # too underdetermined / slow
    assign = {}
    answers = set()
    cap = [3000]

    # Precompute, for each example, the index in `order` after which it's fully assignable.
    def ready(l, r, k):
        # all symbols in this example are within first k of order
        for ch in l[0] + l[1] + l[3] + l[4] + r:
            if ch not in assign:
                return False
        return True

    pos = {ch: i for i, ch in enumerate(order)}

    def check_ready_examples(just_assigned_idx):
        for (l, r) in exs:
            maxpos = max(pos[ch] for ch in l[0] + l[1] + l[3] + l[4] + r)
            if maxpos != just_assigned_idx:
                continue
            a = assign[l[0]] * 10 + assign[l[1]]
            b = assign[l[3]] * 10 + assign[l[4]]
            val = op(a, b)
            inv = {}
            for s, d in assign.items():
                inv.setdefault(d, s)
            if _render(val, inv) != r:
                return False
        return True

    def bt(i):
        if cap[0] <= 0:
            return
        if i == len(order):
            inv = {}
            for s, d in assign.items():
                inv.setdefault(d, s)
            a = assign[q[0]] * 10 + assign[q[1]]
            b = assign[q[3]] * 10 + assign[q[4]]
            r = _render(op(a, b), inv)
            if r is not None:
                answers.add(r)
                cap[0] -= 1
            return
        ch = order[i]
        for d in range(10):
            assign[ch] = d
            if check_ready_examples(i):
                bt(i + 1)
            del assign[ch]
            if cap[0] <= 0:
                return

    bt(0)
    return answers


def solve(prompt):
    exs, q = parse(prompt)
    if not exs or q is None:
        return None
    # concat shortcut
    if all(r == l[:2] + l[3:] for l, r in exs):
        return q[:2] + q[3:]
    if all(r == l[3:] + l[:2] for l, r in exs):
        return q[3:] + q[:2]
    # arithmetic: require a UNIQUE answer across consistent solutions for a single op family
    all_answers = set()
    for opname in OPS:
        ans = _solve_op(exs, q, opname)
        all_answers |= ans
        if len(all_answers) > 1:
            break
    if len(all_answers) == 1:
        return next(iter(all_answers))
    return None
