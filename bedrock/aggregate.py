"""Deterministic aggregation of all per-model Bedrock recon results -> yield_table.md.

Reads every bedrock/<label>/results.jsonl and computes, per model x category:
  pass@1   = mean correctness over all generations
  pass@16  = fraction of PROBLEMS with >=1 correct generation (any-correct)
  n_problems, n_gens, mean output tokens, error rate.
The manager teammate RUNS this script and narrates its output; it never sums by hand.
"""
import os, json, glob
from collections import defaultdict


def main():
    runs = sorted(glob.glob("bedrock/*/results.jsonl"))
    # label -> category -> {problem_id -> [correct,...]}, plus token/err accumulators
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    toks = defaultdict(lambda: defaultdict(list))
    errs = defaultdict(int); calls = defaultdict(int)
    for path in runs:
        label = path.split("/")[1]
        if label.startswith("_test"):
            continue
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            calls[label] += 1
            if r.get("error"):
                errs[label] += 1
                continue
            data[label][r["category"]][r["id"]].append(bool(r.get("correct")))
            if r.get("out_tokens"):
                toks[label][r["category"]].append(r["out_tokens"])

    cats = ["bit_manipulation", "cryptarithm_deduce", "cryptarithm_guess", "equation_numeric_guess"]
    lines = ["# Bedrock recon yield table", "",
             "pass@1 = mean correctness over gens; pass@16 = fraction of problems with >=1 correct gen",
             ""]
    header = "| model | category | n_prob | n_gen | pass@1 | pass@16 | tok_p50 |"
    sep = "|---|---|---|---|---|---|---|"
    for label in sorted(data):
        lines += [f"## {label}  (calls={calls[label]}, errors={errs[label]})", "", header, sep]
        for cat in cats:
            probs = data[label].get(cat, {})
            if not probs:
                continue
            allc = [c for lst in probs.values() for c in lst]
            n_prob = len(probs)
            n_gen = len(allc)
            p1 = sum(allc) / n_gen if n_gen else 0.0
            p16 = sum(1 for lst in probs.values() if any(lst)) / n_prob if n_prob else 0.0
            tl = sorted(toks[label].get(cat, []))
            p50 = tl[len(tl) // 2] if tl else 0
            lines.append(f"| {label} | {cat} | {n_prob} | {n_gen} | {p1:.3f} | {p16:.3f} | {p50} |")
        lines.append("")
    out = "\n".join(lines)
    with open("bedrock/yield_table.md", "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
