"""G0.5: load the 30B base under vLLM with the reference LoRA adapter and confirm
it loads at rank<=32 (judge config) and emits a \\boxed{} answer. Validates the
entire inference path before we train anything.
"""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.grader import is_correct, extract_answer

BASE = "foundation/model"
ADAPTER = "foundation/datasets/dgxchen_trained-adapter"
COT = "foundation/datasets/dgxchen_nemotron-cot-tong/problem_ids_matched.csv"
PROMPT_SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

# pick a few held-out problems across categories for the smoke test
heldout = set(open("eval/heldout_ids.txt").read().split())
samples = []
seen_cat = set()
with open(COT, newline="") as f:
    r = csv.DictReader(f)
    idk = next(k for k in r.fieldnames if k.lstrip("﻿").lower() == "id")
    for row in r:
        if row[idk] in heldout and row["type"] not in seen_cat:
            samples.append({"id": row[idk], "prompt": row["prompt"], "answer": row["answer"], "type": row["type"]})
            seen_cat.add(row["type"])
        if len(seen_cat) >= 5:
            break
print(f"smoke samples: {[s['type'] for s in samples]}", flush=True)

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

print("loading base under vLLM (enable_lora, max_lora_rank=32)...", flush=True)
llm = LLM(
    model=BASE,
    trust_remote_code=True,
    max_model_len=8192,
    gpu_memory_utilization=0.85,
    enable_lora=True,
    max_lora_rank=32,
    dtype="bfloat16",
)
tok = llm.get_tokenizer()

def build(prompt):
    msgs = [{"role": "user", "content": prompt + PROMPT_SUFFIX}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

prompts = [build(s["prompt"]) for s in samples]
sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=4096)
lr = LoRARequest("ref", 1, ADAPTER)

print("generating (greedy, with adapter)...", flush=True)
outs = llm.generate(prompts, sp, lora_request=lr)
n_ok = 0
for s, o in zip(samples, outs):
    text = o.outputs[0].text
    has_box = "\\boxed{" in text
    correct = is_correct(text, s["answer"])
    n_ok += correct
    print(f"[{s['type']}] boxed={has_box} extract={extract_answer(text)!r} gt={s['answer']!r} correct={correct} tok={len(o.outputs[0].token_ids)}", flush=True)
print(f"\nG0.5: adapter loaded under judge-config vLLM and generated; {n_ok}/{len(samples)} correct on smoke set", flush=True)
print("G0.5_DONE", flush=True)
