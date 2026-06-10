"""Score a trained adapter on the stratified held-out split (greedy, judge config).

Loads base + adapter under vLLM with the exact judge sampling params, generates on
every held-out problem (deduped by id), grades per-category + overall.

Usage: python eval/score_heldout.py --adapter artifacts/parity_v1/adapter --run parity_v1
"""
import os, sys, json, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.grader import is_correct, extract_answer

BASE = "foundation/model"
COT = "foundation/datasets/dgxchen_nemotron-cot-tong/problem_ids_matched.csv"
SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--run", default="eval")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    heldout = set(open("eval/heldout_ids.txt").read().split())
    probs, seen = [], set()
    with open(COT, newline="") as f:
        r = csv.DictReader(f)
        idk = next(k for k in r.fieldnames if k.lstrip("﻿").lower() == "id")
        for row in r:
            i = row[idk]
            if i in heldout and i not in seen:
                seen.add(i)
                probs.append({"id": i, "prompt": row["prompt"], "answer": row["answer"], "type": row["type"]})
    if args.limit:
        probs = probs[: args.limit]
    print(f"held-out problems (deduped): {len(probs)}", flush=True)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=BASE, trust_remote_code=True, max_model_len=8192,
              gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=32, dtype="bfloat16")
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": p["prompt"] + SUFFIX}],
                                       tokenize=False, add_generation_prompt=True) for p in probs]
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=7680)
    lr = LoRARequest(args.run, 1, args.adapter)
    print("generating greedy over held-out...", flush=True)
    outs = llm.generate(prompts, sp, lora_request=lr)

    from collections import defaultdict
    by = defaultdict(lambda: [0, 0])
    rows = []
    for p, o in zip(probs, outs):
        text = o.outputs[0].text
        ok = is_correct(text, p["answer"])
        by[p["type"]][1] += 1
        by[p["type"]][0] += ok
        rows.append({"id": p["id"], "type": p["type"], "correct": bool(ok),
                     "extracted": extract_answer(text), "answer": p["answer"],
                     "tokens": len(o.outputs[0].token_ids)})
    os.makedirs(f"eval/results", exist_ok=True)
    with open(f"eval/results/heldout_{args.run}.jsonl", "w") as f:
        for r_ in rows: f.write(json.dumps(r_) + "\n")

    tot_c = sum(v[0] for v in by.values()); tot_n = sum(v[1] for v in by.values())
    print(f"\n=== HELD-OUT SCORE: {args.run} ===", flush=True)
    print(f"{'category':<24}{'correct/total':>14}{'rate':>8}")
    for c in sorted(by):
        cc, nn = by[c]
        print(f"{c:<24}{f'{cc}/{nn}':>14}{100*cc/nn:>7.1f}%")
    print(f"{'OVERALL':<24}{f'{tot_c}/{tot_n}':>14}{100*tot_c/tot_n:>7.1f}%")
    print(f"HELDOUT_RESULT {json.dumps({'run':args.run,'overall':round(tot_c/tot_n,4),'by':{c:[by[c][0],by[c][1]] for c in by}})}", flush=True)


if __name__ == "__main__":
    main()
