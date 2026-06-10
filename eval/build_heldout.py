"""Build a per-category stratified held-out split from the CoT training CSV.

The CoT data (problem_ids_matched.csv) carries the 9 canonical category labels in
`type`, so no 7-vs-9 reconciliation is needed. We hold out a stratified fraction per
category for local eval; these ids are NEVER trained on. Writes:
  eval/heldout_ids.txt   - held-out problem ids (one per line)
  eval/train_ids.txt     - complementary training ids
  eval/split_report.txt  - per-category counts + disjointness assertion
"""
from __future__ import annotations
import csv, os, random
from collections import defaultdict

SRC = "foundation/datasets/dgxchen_nemotron-cot-tong/problem_ids_matched.csv"
OUT = "eval"
SEED = 42
HOLDOUT_FRAC = 0.15
MIN_PER_CAT = 15   # ensure at least this many held out where the category allows
os.makedirs(OUT, exist_ok=True)


def load_rows(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        # first column may carry a BOM; normalize the id key
        id_key = next(k for k in r.fieldnames if k.lstrip("﻿").lower() == "id")
        rows = []
        for row in r:
            rows.append({"id": row[id_key], "type": row["type"]})
        return rows


def main():
    rows = load_rows(SRC)
    # Split on UNIQUE problem ids (some ids have multiple CoT rows, up to 14x).
    # id -> type is 1:1, so dedup ids per category to avoid leakage.
    seen = set()
    by_cat = defaultdict(list)
    for row in rows:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        by_cat[row["type"]].append(row["id"])
    n_unique = len(seen)

    rng = random.Random(SEED)
    heldout, train = [], []
    report = ["category            total  heldout  train"]
    for cat in sorted(by_cat):
        ids = by_cat[cat][:]
        rng.shuffle(ids)
        n_hold = max(min(MIN_PER_CAT, len(ids) // 2), int(round(len(ids) * HOLDOUT_FRAC)))
        n_hold = min(n_hold, len(ids))
        h, t = ids[:n_hold], ids[n_hold:]
        heldout += h
        train += t
        report.append(f"{cat:<18} {len(ids):>6} {len(h):>8} {len(t):>6}")

    hset, tset = set(heldout), set(train)
    assert hset.isdisjoint(tset), "held-out and train id sets overlap!"
    assert len(hset) + len(tset) == n_unique, "id partition does not cover all unique ids"

    with open(f"{OUT}/heldout_ids.txt", "w") as f:
        f.write("\n".join(heldout) + "\n")
    with open(f"{OUT}/train_ids.txt", "w") as f:
        f.write("\n".join(train) + "\n")
    report.append("-" * 40)
    report.append(f"{'TOTAL':<18} {n_unique:>6} {len(heldout):>8} {len(train):>6}")
    report.append("disjoint: True   covers_all: True")
    text = "\n".join(report)
    with open(f"{OUT}/split_report.txt", "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
