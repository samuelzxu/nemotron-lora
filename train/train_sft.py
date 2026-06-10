"""Parity training (US-007): replicate the dgxchen unsloth 0.85 recipe locally.

Trains on the winner's CoT (problem_ids_matched.csv) MINUS held-out ids, with the exact
proven hyperparameters. import unsloth FIRST. Saves adapter to artifacts/<run-name>/.

Usage:
  python train/train_sft.py --run parity_v1 [--epochs 1 --lr 2e-4 --rank 32 --alpha 32]
"""
import unsloth  # MUST be first
import argparse, csv, math, os, random, re, time
from collections import defaultdict
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Sampler
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
from unsloth import FastLanguageModel


class SaveEpochAdapter(TrainerCallback):
    """Snapshot a clean PEFT adapter at the end of each epoch -> artifacts/<run>/adapter_ep{N}/.

    Writes a `.done` sentinel last so an external watcher only picks up fully-saved adapters
    (enables auto-submit of the epoch-1 checkpoint the moment it lands, mid 2-epoch run).
    """
    def __init__(self, out_dir, tokenizer):
        self.out_dir = out_dir
        self.tok = tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kw):
        ep = int(round(state.epoch or 0))
        if ep < 1:
            return
        d = f"{self.out_dir}/adapter_ep{ep}"
        os.makedirs(d, exist_ok=True)
        model.save_pretrained(d)
        self.tok.save_pretrained(d)
        with open(f"{d}/.done", "w") as fh:
            fh.write(f"epoch={ep} step={state.global_step}\n")
        print(f"EPOCH_ADAPTER_SAVED {d}", flush=True)

BASE = "foundation/model"
COT = "foundation/datasets/dgxchen_nemotron-cot-tong/problem_ids_matched.csv"
SEED = 42
MAX_SEQ = 8192
PROMPT_SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"


def build_stratified_index_order(labels, batch_size, seed):
    by_label = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)
    rng = random.Random(seed)
    for lst in by_label.values():
        rng.shuffle(lst)
    n_batches = max(1, math.ceil(len(labels) / batch_size))
    batches = [[] for _ in range(n_batches)]
    order = list(range(n_batches)); rng.shuffle(order)
    assigned = 0
    for label in sorted(by_label):
        for idx in by_label[label]:
            batches[order[assigned % n_batches]].append(idx); assigned += 1
    out = [i for b in batches for i in b]
    assert len(out) == len(labels)
    return out


class OrderSampler(Sampler):
    def __init__(self, order): self.order = list(order)
    def __iter__(self): return iter(self.order)
    def __len__(self): return len(self.order)


class StratifiedSFTTrainer(SFTTrainer):
    def __init__(self, *a, stratified_order=None, **k):
        super().__init__(*a, **k); self.stratified_order = stratified_order
    def get_train_dataloader(self):
        if self.stratified_order is None:
            return super().get_train_dataloader()
        kw = dict(batch_size=self.args.per_device_train_batch_size,
                  sampler=OrderSampler(self.stratified_order),
                  collate_fn=self.data_collator,
                  num_workers=self.args.dataloader_num_workers,
                  pin_memory=self.args.dataloader_pin_memory,
                  drop_last=self.args.dataloader_drop_last)
        return DataLoader(self.train_dataset, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="parity_v1")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--scheduler", default="linear")
    ap.add_argument("--warmup-ratio", type=float, default=0.0)
    ap.add_argument("--freeze-experts", action="store_true",
                    help="exclude routed experts from LoRA (regularizer A/B)")
    ap.add_argument("--extra-data", default=None,
                    help="optional JSONL of extra {id,prompt,answer,type,generated_cot} records to append")
    ap.add_argument("--extra-data-mult", type=int, default=1,
                    help="repeat each extra-data record N times (oversampling)")
    ap.add_argument("--full-data", action="store_true",
                    help="train on the FULL COT dataset (do NOT exclude the held-out split)")
    args = ap.parse_args()
    out_dir = f"artifacts/{args.run}"
    os.makedirs(out_dir, exist_ok=True)

    if args.full_data:
        train_ids = None
        print("FULL dataset: held-out NOT excluded", flush=True)
    else:
        train_ids = set(open("eval/train_ids.txt").read().split())  # held-out EXCLUDED by construction
        print(f"train ids (held-out excluded): {len(train_ids)}", flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE, max_seq_length=MAX_SEQ, load_in_4bit=False, load_in_8bit=False,
        full_finetuning=False, trust_remote_code=True, unsloth_force_compile=False,
        attn_implementation="eager", dtype=torch.bfloat16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "up_proj", "down_proj"],
        bias="none", use_gradient_checkpointing="unsloth", random_state=SEED,
    )
    model.print_trainable_parameters()

    records, rtypes = [], []
    with open(COT, newline="") as f:
        r = csv.DictReader(f)
        idk = next(k for k in r.fieldnames if k.lstrip("﻿").lower() == "id")
        for row in r:
            if train_ids is not None and row[idk] not in train_ids:
                continue
            cot = str(row["generated_cot"])
            if not cot or cot == "nan" or len(cot.strip()) < 5:
                continue
            cot_clean = re.sub(r"\\boxed\{[^}]*\}", "", cot).rstrip()
            user = str(row["prompt"]) + PROMPT_SUFFIX
            asst = cot_clean + f"\n</think>\n\\boxed{{{row['answer']}}}"
            records.append({"messages": [{"role": "user", "content": user},
                                         {"role": "assistant", "content": asst}]})
            rtypes.append(str(row["type"]))
    if args.extra_data and os.path.exists(args.extra_data):
        import json as _json
        n0 = len(records)
        for line in open(args.extra_data):
            line = line.strip()
            if not line:
                continue
            e = _json.loads(line)
            cot = str(e["generated_cot"])
            cot_clean = re.sub(r"\\boxed\{[^}]*\}", "", cot).rstrip()
            user = str(e["prompt"]) + PROMPT_SUFFIX
            asst = cot_clean + f"\n</think>\n\\boxed{{{e['answer']}}}"
            rec_msg = {"messages": [{"role": "user", "content": user},
                                    {"role": "assistant", "content": asst}]}
            for _ in range(args.extra_data_mult):
                records.append(rec_msg)
                rtypes.append(str(e["type"]))
        print(f"extra-data: +{len(records)-n0} records from {args.extra_data} (x{args.extra_data_mult})", flush=True)
    print(f"SFT records: {len(records)}", flush=True)
    dataset = HFDataset.from_list(records)

    def fmt(example):
        out = []
        msgs = example["messages"]
        convs = [msgs] if (msgs and isinstance(msgs[0], dict)) else msgs
        for c in convs:
            try:
                t = tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False, enable_thinking=True)
            except TypeError:
                t = tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
            out.append(t)
        return out

    cfg = SFTConfig(
        output_dir=f"{out_dir}/sft_output", num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type=args.scheduler, warmup_ratio=args.warmup_ratio, max_length=MAX_SEQ,
        adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-8, weight_decay=0.0, max_grad_norm=1e9,
        logging_steps=10, save_strategy="no", bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, dataloader_num_workers=2,
        remove_unused_columns=False, seed=SEED, report_to="none", packing=False,
    )
    eff = max(1, cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)
    order = build_stratified_index_order(rtypes, eff, SEED)

    trainer = StratifiedSFTTrainer(model=model, args=cfg, train_dataset=dataset,
                                   processing_class=tokenizer, formatting_func=fmt,
                                   stratified_order=order,
                                   callbacks=[SaveEpochAdapter(out_dir, tokenizer)])
    print("=== training start ===", flush=True)
    t0 = time.time(); trainer.train()
    print(f"=== training done in {(time.time()-t0)/60:.1f} min ===", flush=True)

    adapter_dir = f"{out_dir}/adapter"
    model.save_pretrained(adapter_dir); tokenizer.save_pretrained(adapter_dir)
    print(f"ADAPTER_SAVED {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
