#!/usr/bin/env bash
set -uo pipefail
export PATH="/home/ec2-user/.local/bin:/usr/local/bin:/usr/bin:$PATH"
source ~/.claudeinit 2>/dev/null
cd /home/ec2-user/dev/nemotron_training/foundation
log(){ echo "[$(date +%H:%M:%S)] $*"; }

DS="datasets"
for ref in mayukh18/nemotron-packages dgxchen/nemotron-cot-tong dennisfong/nvidia-nemotron-offline-packages rubyducklove/nvidia-cutlass dgxchen/trained-adapter; do
  out="$DS/$(echo "$ref" | tr '/' '_')"
  mkdir -p "$out"
  log "Downloading dataset $ref -> $out"
  kaggle datasets download "$ref" -p "$out" --unzip 2>&1 | tail -2
  log "done $ref ; size: $(du -sh "$out" 2>/dev/null | cut -f1)"
done

log "Downloading base model nemotron-3-nano-30b-a3b-bf16 (LARGE ~60GB)"
kaggle models instances versions download metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1 -p model --untar 2>&1 | tail -3 \
  || kaggle models instances versions download metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1 -p model 2>&1 | tail -3
log "model dir size: $(du -sh model 2>/dev/null | cut -f1)"
log "ALL DOWNLOADS COMPLETE"
df -h /home/ec2-user | tail -1
