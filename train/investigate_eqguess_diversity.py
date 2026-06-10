"""Investigate whether 2-3 DIVERSE sound CoTs per problem actually exist in the eq_guess captures.

For each problem: collect all exact-string-correct, roughly-in-budget gens across the 3 models,
then greedily cluster by text similarity to count *distinct reasoning approaches* (not near-dups).
Prints distribution + sample side-by-sides so we can eyeball real diversity vs restatement.
"""
import json, os, re, sys
from collections import defaultdict
from difflib import SequenceMatcher
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.grader import extract_answer

SAMPLE = "bedrock/eqguess_sample.jsonl"
CAP = {"deepseek": "bedrock/eqguess_capture/eq_deepseek/results.jsonl",
       "gptoss":   "bedrock/eqguess_capture/eq_gptoss/results.jsonl",
       "mistral":  "bedrock/eqguess_capture/eq_mistral/results.jsonl"}
MAX_TOK = 8192
SIM_THRESH = 0.55   # >this similarity => "same approach"; tune by eyeballing samples


def num_eq(pred, gt):
    if pred is None:
        return False
    ps, gs = str(pred).strip(), str(gt).strip()
    if ps.lower() == gs.lower():
        return True
    try:
        return float(ps) == float(gs)
    except Exception:
        return False


def norm(t):
    return re.sub(r"\s+", " ", t.lower()).strip()


def distinct_approaches(cots):
    """Greedy cluster: return list of representative indices, one per distinct approach."""
    reps = []
    normed = [norm(c) for c in cots]
    for i, n in enumerate(normed):
        if all(SequenceMatcher(None, n, normed[j]).quick_ratio() < SIM_THRESH for j in reps):
            reps.append(i)
    return reps


def main():
    held = set(open("eval/heldout_ids.txt").read().split())
    probs = {}
    for l in open(SAMPLE):
        r = json.loads(l)
        if r["id"] in held:
            continue
        probs[r["id"]] = {"prompt": r["prompt"], "answer": str(r["answer"]), "status": r.get("status")}

    # collect exact-correct in-budget gens per problem: id -> list of (model, cot)
    bag = defaultdict(list)
    for src, path in CAP.items():
        for l in open(path):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("error"):
                continue
            pid, gt = r.get("id"), str(r.get("answer"))
            if pid not in probs:
                continue
            cot = (r.get("text") or r.get("answer_text") or "").strip()
            if not cot:
                continue
            tok = r.get("out_tokens") or 0
            if tok and tok > MAX_TOK:
                continue
            if not num_eq(extract_answer(cot), gt):
                continue
            bag[pid].append((src, cot))

    # per-problem stats
    rows = []
    for pid, gens in bag.items():
        reps = distinct_approaches([c for _, c in gens])
        models = sorted({s for s, _ in gens})
        rep_models = sorted({gens[i][0] for i in reps})
        rows.append((pid, probs[pid]["status"], len(gens), len(reps), models, rep_models))

    rows.sort(key=lambda r: (-r[3], -r[2]))
    print(f"problems with >=1 exact-correct in-budget gen: {len(rows)}")
    # distribution of distinct-approach counts
    dist = defaultdict(int)
    for _, _, _, nrep, _, _ in rows:
        dist[min(nrep, 3)] += 1
    print("distinct-approach count (capped at 3+):", dict(sorted(dist.items())))
    by_status = defaultdict(lambda: [0, 0])  # status -> [problems, problems with >=2 distinct]
    for _, st, _, nrep, _, _ in rows:
        by_status[st][0] += 1
        if nrep >= 2:
            by_status[st][1] += 1
    print("by status [problems, with>=2 distinct]:", {k: v for k, v in by_status.items()})
    print(f"\n{'id':<10}{'status':<18}{'gens':>5}{'distinct':>9}  models_with_distinct")
    for pid, st, ng, nrep, models, rep_models in rows[:40]:
        print(f"{pid:<10}{st:<18}{ng:>5}{nrep:>9}  {','.join(rep_models)}")

    # show side-by-side of 2 hard problems with >=2 distinct approaches
    shown = 0
    for pid, st, ng, nrep, models, rep_models in rows:
        if nrep >= 2 and st == "rule_unknown" and shown < 2:
            shown += 1
            gens = bag[pid]
            reps = distinct_approaches([c for _, c in gens])
            print(f"\n{'='*80}\nPROBLEM {pid}  status={st}  answer={probs[pid]['answer']}")
            print("PROMPT:", probs[pid]["prompt"].replace("\n", " | "))
            for k, ri in enumerate(reps[:3]):
                src, cot = gens[ri]
                print(f"\n--- approach {k+1} [{src}] ---")
                print(cot[:700])


if __name__ == "__main__":
    main()
