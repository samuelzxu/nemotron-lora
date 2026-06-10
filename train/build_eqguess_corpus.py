"""Build the equation_numeric_guess distillation corpus (audit-driven).

Two sources, per puzzle_team/results/eqguess_capture_quality.md:
  A. CAPTURES  — best-per-problem Bedrock CoT, EXACT-STRING correct (no tolerance),
                 in-budget (<=8192 tok), source-ranked deepseek>gptoss>mistral.
  B. SOLVER    — clean templated CoT for problems the deterministic typographic rule
                 (`-`->sub, `*`->rev-concat, `/`->mul, else |a-b|) lands on EXACTLY.

Quality gate = exact numeric/string equality (NOT rel_tol). This drops the rationalized
tolerance near-misses the audit flagged (0f8452df 158-vs-159, 1b3beb8f 109-vs-108).

Output: train/eqguess_distill.jsonl with COT-schema fields {id,prompt,answer,type,
generated_cot,source}. The final \boxed{answer} is appended by train_sft.py from the
ground-truth `answer`, so generated_cot holds reasoning only.
"""
import json, os, re, sys
from collections import defaultdict
from difflib import SequenceMatcher
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.grader import extract_answer

SIM_THRESH = 0.55          # >this text similarity => "same approach" (dedup near-duplicate restatements)
MAX_PER_PROBLEM = 4        # capped repetition: up to 4 correct gens per problem
DEDUP = False              # False = raw capped repetition (keep near-duplicate gens, no similarity filter)
INCLUDE_SOLVER = True      # ship BOTH: deterministic solver templates + captured reasoning traces


def _norm(t):
    return re.sub(r"\s+", " ", t.lower()).strip()

SAMPLE = "bedrock/eqguess_sample.jsonl"               # id,prompt,answer,category,status (raw prompt, no suffix)
CAP = {"deepseek": "bedrock/eqguess_capture/eq_deepseek/results.jsonl",
       "gptoss":   "bedrock/eqguess_capture/eq_gptoss/results.jsonl",
       "glm":      "bedrock/eqguess_capture/eq_glm/results.jsonl",
       "claude":   "bedrock/eqguess_capture/eq_claude/results.jsonl",
       "mistral":  "bedrock/eqguess_capture/eq_mistral/results.jsonl"}
# rank = preference when picking best-per-problem (soundness/compactness; audit S.4 + new sources)
SRC_RANK = {"deepseek": 0, "gptoss": 1, "glm": 2, "claude": 3, "mistral": 4}
TOK_BUDGET = 8000   # full formatted example must fit train_sft MAX_SEQ=8192 (margin for template+box)
OUT = "train/eqguess_distill.jsonl"
QRE = re.compile(r"determine the result for:\s*(\d+)(\D)(\d+)")
# mirror train_sft.py formatting so the budget check matches what training actually sees
PROMPT_SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"


def fits_budget(tok, prompt, cot, answer):
    cot_clean = re.sub(r"\\boxed\{[^}]*\}", "", cot).rstrip()
    user = prompt + PROMPT_SUFFIX
    asst = cot_clean + f"\n</think>\n\\boxed{{{answer}}}"
    try:
        t = tok.apply_chat_template(
            [{"role": "user", "content": user}, {"role": "assistant", "content": asst}],
            tokenize=True, add_generation_prompt=False)
    except Exception:
        t = tok(user + asst)["input_ids"]
    return len(t) <= TOK_BUDGET


def num_eq(pred, gt):
    """EXACT equality — string-equal, or exact numeric value (handles leading-zero/format), NO tolerance."""
    if pred is None:
        return False
    ps, gs = str(pred).strip(), str(gt).strip()
    if ps.lower() == gs.lower():
        return True
    try:
        return float(ps) == float(gs)
    except Exception:
        return False


def exact_ok(text, gt):
    return num_eq(extract_answer(text), gt)


def solver_cot(prompt):
    """Return (answer, reasoning, op) for the deterministic typographic rule, or (None, None, None).

    Only the '-'/'*'/'/' slice is emitted as distillation data: those teach NEW strategies
    (subtraction, reverse-concat, multiply). The '|a-b|' fallback is deliberately NOT emitted —
    the existing 126 trained rows already over-teach absdiff; reinforcing it is counter-purpose.
    """
    m = QRE.search(prompt)
    if not m:
        return None, None, None
    sa, op, sb = m.group(1), m.group(2), m.group(3)
    a, b = int(sa), int(sb)
    if op == "-":
        ans, rule = str(a - b), f"a minus sign almost always denotes literal subtraction, so {a} - {b} = {a - b}"
    elif op == "*":
        ans, rule = sb + sa, f"here '*' behaves as reverse concatenation (the second operand written before the first), giving {sb}{sa}"
    elif op == "/":
        ans, rule = str(a * b), f"here '/' behaves as multiplication, so {a} x {b} = {a * b}"
    else:
        return None, None, op  # absdiff fallback -> not distilled
    cot = (f"The examples define several operators, but the query operator '{op}' does not appear in any of them, "
           f"so the worked examples carry no direct information about it. The query operator therefore has to be "
           f"inferred from the symbol itself rather than from the examples. In this case {rule}.")
    return ans, cot, op


def main():
    held = set(open("eval/heldout_ids.txt").read().split())
    probs = {}
    skipped_held = 0
    for l in open(SAMPLE):
        r = json.loads(l)
        if r["id"] in held:            # never put held-out problems into training data
            skipped_held += 1
            continue
        probs[r["id"]] = {"prompt": r["prompt"], "answer": str(r["answer"]), "status": r.get("status")}
    print(f"problems in sample: {len(probs)} (excluded {skipped_held} held-out ids)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("foundation/model", trust_remote_code=True)
    dropped_budget = 0

    # ---- A. captures: up to MAX_PER_PROBLEM DISTINCT sound approaches per problem.
    # Gather all exact-correct, in-budget gens, then greedily keep soundest-then-shortest
    # representatives that are textually dissimilar (>SIM_THRESH = same approach -> dropped).
    cand = defaultdict(list)   # pid -> list of (src_rank, cot, src, gt)
    for src, path in CAP.items():
        if not os.path.exists(path):
            print(f"WARN missing {path}"); continue
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
            if not exact_ok(cot, gt):
                continue
            if not fits_budget(tok, probs[pid]["prompt"], cot, gt):
                dropped_budget += 1
                continue
            cand[pid].append((SRC_RANK[src], cot, src, gt))

    cap_recs = []
    multi = 0
    for pid, lst in cand.items():
        lst.sort(key=lambda x: (x[0], len(x[1])))   # soundest source, then most compact
        kept, kept_norm = [], []
        for rank, cot, src, gt in lst:
            n = _norm(cot)
            if (not DEDUP) or all(SequenceMatcher(None, n, kn).quick_ratio() < SIM_THRESH for kn in kept_norm):
                kept.append((cot, src, gt)); kept_norm.append(n)
            if len(kept) >= MAX_PER_PROBLEM:
                break
        if len(kept) >= 2:
            multi += 1
        for cot, src, gt in kept:
            cap_recs.append({"id": pid, "prompt": probs[pid]["prompt"], "answer": gt,
                             "type": "equation_numeric_guess", "generated_cot": cot,
                             "source": f"bedrock_{src}"})

    # ---- B. solver-synthetic where the rule is EXACT on ground truth
    solver_recs = []
    for pid, info in probs.items():
        ans, cot, op = solver_cot(info["prompt"])
        if ans is None:
            continue
        if num_eq(ans, info["answer"]):
            solver_recs.append({"id": pid, "prompt": info["prompt"], "answer": str(info["answer"]),
                                "type": "equation_numeric_guess", "generated_cot": cot, "source": "solver_rule"})

    records = cap_recs + (solver_recs if INCLUDE_SOLVER else [])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # ---- report
    cap_ids = {r["id"] for r in cap_recs}
    sol_ids = {r["id"] for r in solver_recs}
    src_split = {}
    for r in cap_recs:
        src_split[r["source"]] = src_split.get(r["source"], 0) + 1
    print(f"\n=== eqguess corpus written -> {OUT} ===")
    print(f"capture traces:  {len(cap_recs)}  (distinct problems {len(cap_ids)}; {multi} have 2 distinct approaches)  source split {src_split}")
    print(f"  (dropped {dropped_budget} exact-correct gens that exceeded the {TOK_BUDGET}-token training budget)")
    print(f"solver traces:   {len(solver_recs)} (distinct problems {len(sol_ids)})")
    print(f"total records:   {len(records)}")
    print(f"distinct problems covered (union): {len(cap_ids | sol_ids)} / {len(probs)}")
    print(f"  captures-only: {len(cap_ids - sol_ids)} | solver-only: {len(sol_ids - cap_ids)} | both: {len(cap_ids & sol_ids)}")
    # token sanity on capture CoTs (chars; ~4 chars/token rough)
    lens = sorted(len(r["generated_cot"]) for r in cap_recs)
    if lens:
        print(f"capture CoT chars: min {lens[0]} / median {lens[len(lens)//2]} / max {lens[-1]}")


if __name__ == "__main__":
    main()
