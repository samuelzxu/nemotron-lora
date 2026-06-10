"""Shared solve-rate harness for the puzzle-cracking team.

A solver is a Python module exposing `solve(prompt: str) -> str | None`.
The harness feeds ONLY the prompt (never the answer) and grades the returned
answer against ground truth, reporting overall solve rate + a breakdown by the
winner's status (rule_unknown = problems the winner could NOT solve — that's where
the roof improvement lives).

Usage:
  python harness.py <category> <solver_module.py> [--limit N] [--timeout SEC]
Categories: cryptarithm_deduce cryptarithm_guess equation_numeric_guess
            bit_manipulation equation_numeric_deduce
"""
import sys, os, json, argparse, importlib.util, signal
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.grader import is_correct

DATA = os.path.join(os.path.dirname(__file__), "data")


class _Timeout(Exception): pass
def _alarm(signum, frame): raise _Timeout()


def load_solver(path):
    spec = importlib.util.spec_from_file_location("solver_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "solve"), "solver module must define solve(prompt)->str"
    return mod.solve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("category")
    ap.add_argument("solver")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=5.0, help="per-problem seconds")
    ap.add_argument("--status", default=None, help="filter to a winner_status (e.g. rule_unknown)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(f"{DATA}/{args.category}.jsonl")]
    if args.status:
        rows = [r for r in rows if r.get("winner_status") == args.status]
    if args.limit:
        rows = rows[: args.limit]
    solve = load_solver(args.solver)

    signal.signal(signal.SIGALRM, _alarm)
    by_status = defaultdict(lambda: [0, 0])  # status -> [correct, total]
    correct = err = timeouts = 0
    for r in rows:
        st = r.get("winner_status", "?")
        by_status[st][1] += 1
        try:
            signal.setitimer(signal.ITIMER_REAL, args.timeout)
            ans = solve(r["prompt"])
            signal.setitimer(signal.ITIMER_REAL, 0)
            ok = ans is not None and is_correct(str(ans), r["answer"])
        except _Timeout:
            signal.setitimer(signal.ITIMER_REAL, 0); ok = False; timeouts += 1
        except Exception:
            signal.setitimer(signal.ITIMER_REAL, 0); ok = False; err += 1
        if ok:
            correct += 1; by_status[st][0] += 1

    n = len(rows)
    print(f"\n=== {args.category} | solver={os.path.basename(args.solver)} ===")
    print(f"SOLVE RATE: {correct}/{n} = {100*correct/n:.1f}%   (errors={err}, timeouts={timeouts})")
    print("by winner_status (correct/total) -- rule_unknown = NEW ground gained:")
    for st in sorted(by_status):
        c, t = by_status[st]
        print(f"  {st:<18} {c}/{t} = {100*c/t:.1f}%")
    # machine-readable line for the manager to aggregate
    print(f"RESULT_JSON {json.dumps({'category':args.category,'solver':os.path.basename(args.solver),'correct':correct,'total':n,'rate':round(correct/n,4),'by_status':{k:v for k,v in by_status.items()}})}")


if __name__ == "__main__":
    main()
