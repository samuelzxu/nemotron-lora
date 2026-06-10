"""Generic Bedrock recon driver (shared protocol for all 8 models).

For each problem in the sample x N_GENS, call the Converse API, extract the answer,
grade against ground truth, and record correctness + token length. Concurrent + resumable
(checkpoints to results.jsonl; reruns skip completed (id, gen) pairs).

Per-model specialization is passed via CLI/env (model_id, thinking config, max_tokens, temp).
Each teammate runs ONE model with the SAME sample/grader/schema so the manager can aggregate.

Usage:
  python driver.py --model-id deepseek.v3.2 --label deepseek_v3p2 \
      --max-tokens 8192 --n-gens 16 --temperature 0.8 --concurrency 8 \
      [--thinking '{"reasoning_config":{"type":"enabled","budget_tokens":4096}}']
"""
import os, sys, json, argparse, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.grader import is_correct, extract_answer

import boto3
from botocore.config import Config

SUFFIX = "\nPlease put your final answer inside \\boxed{}. For example: \\boxed{your answer}"
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def load_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                done.add((r["id"], r["gen"]))
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--sample", default="bedrock/sample.jsonl")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-gens", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--thinking", default=None, help="JSON for additionalModelRequestFields")
    ap.add_argument("--system", default=None, help="system prompt (e.g. Nemotron 'detailed thinking on')")
    ap.add_argument("--limit", type=int, default=0, help="limit #problems (0=all) for validation runs")
    args = ap.parse_args()

    out_dir = args.out_dir or f"bedrock/{args.label}"
    os.makedirs(out_dir, exist_ok=True)
    results_path = f"{out_dir}/results.jsonl"
    extra = json.loads(args.thinking) if args.thinking else None

    sample = [json.loads(l) for l in open(args.sample)]
    if args.limit:
        sample = sample[: args.limit]
    done = load_done(results_path)

    cfg = Config(retries={"max_attempts": 6, "mode": "adaptive"}, read_timeout=300, connect_timeout=15)
    br = boto3.client("bedrock-runtime", region_name=REGION, config=cfg)
    lock = threading.Lock()
    fout = open(results_path, "a")

    tasks = [(p, g) for p in sample for g in range(args.n_gens) if (p["id"], g) not in done]
    print(f"[{args.label}] model={args.model_id} sample={len(sample)} n_gens={args.n_gens} "
          f"todo={len(tasks)} (done={len(done)})", flush=True)

    def work(task):
        p, g = task
        msgs = [{"role": "user", "content": [{"text": p["prompt"] + SUFFIX}]}]
        # Extended-thinking models (Claude reasoning_config) require temperature=1 and NO topP.
        if extra and isinstance(extra, dict) and "reasoning_config" in extra:
            inf = {"maxTokens": args.max_tokens, "temperature": 1.0}
        else:
            inf = {"maxTokens": args.max_tokens, "temperature": args.temperature, "topP": args.top_p}
        kw = dict(modelId=args.model_id, messages=msgs, inferenceConfig=inf)
        if extra:
            kw["additionalModelRequestFields"] = extra
        if args.system:
            kw["system"] = [{"text": args.system}]
        try:
            r = br.converse(**kw)
            content = r["output"]["message"]["content"]
            txt = "".join(c.get("text", "") for c in content)
            rsn = ""
            for c in content:
                rc = c.get("reasoningContent")
                if isinstance(rc, dict):
                    rsn += rc.get("reasoningText", {}).get("text", "")
            full = (rsn + "\n" + txt) if rsn else txt
            usage = r.get("usage", {})
            ok = bool(is_correct(full, p["answer"]))
            rec = {
                "id": p["id"], "gen": g, "category": p["category"], "status": p.get("status"),
                "correct": ok,
                "extracted": extract_answer(full), "answer": p["answer"],
                "has_box": "\\boxed{" in full,
                "out_tokens": usage.get("outputTokens"), "reasoning_chars": len(rsn),
                "stop": r.get("stopReason"),
                # PERSIST THE TRACE for distillation: keep full text on correct gens
                # (and the answer text on all gens for reproducibility analysis).
                "text": full if ok else None,
                "answer_text": txt,
            }
        except Exception as e:
            rec = {"id": p["id"], "gen": g, "category": p["category"],
                   "error": f"{type(e).__name__}: {str(e)[:160]}"}
        with lock:
            fout.write(json.dumps(rec) + "\n"); fout.flush()
        return rec

    t0 = time.time(); n = 0; errs = 0; ok = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for rec in as_completed([ex.submit(work, t) for t in tasks]):
            r = rec.result(); n += 1
            if r.get("error"):
                errs += 1
            elif r.get("correct"):
                ok += 1
            if n % 50 == 0:
                print(f"[{args.label}] {n}/{len(tasks)} done, ok={ok} err={errs} "
                      f"({(time.time()-t0)/60:.1f}m)", flush=True)
    fout.close()
    print(f"[{args.label}] DONE {n} calls, ok={ok} err={errs} in {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
