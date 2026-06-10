#!/usr/bin/env bash
# Build the version-matched venv from the offline wheels (replicates konbu17/dgxchen env).
# Python 3.12, torch 2.10/cu12, transformers/peft/trl/datasets/accelerate + mamba_ssm/causal_conv1d + vllm.
set -uo pipefail
export PATH="/home/ec2-user/.local/bin:/usr/local/bin:/usr/bin:$PATH"
cd /home/ec2-user/dev/nemotron_training
log(){ echo "[$(date +%H:%M:%S)] $*"; }

PKG1="$PWD/foundation/datasets/mayukh18_nemotron-packages/packages"
PKG2="$PWD/foundation/datasets/dennisfong_nvidia-nemotron-offline-packages/offline_packages"
FL="--find-links $PKG1 --find-links $PKG2"
VENV="$PWD/foundation/.venv"

log "Creating venv (Python 3.12) at $VENV"
uv venv "$VENV" --python 3.12 || { log "venv create FAILED"; exit 1; }
PY="$VENV/bin/python"

log "1) torch 2.10.0 (matches mamba/causal/flash cu12torch2.10 wheels)"
uv pip install --python "$PY" --no-index $FL "torch==2.10.0" 2>&1 | tail -3

log "2) core stack from offline wheels (konbu17 set)"
uv pip install --python "$PY" --no-index $FL \
  transformers peft trl datasets accelerate bitsandbytes 2>&1 | tail -5

log "3) mamba_ssm + causal_conv1d (cp312 torch2.10 cu12 abiTRUE), --no-deps"
CAUSAL=$(ls "$PKG1"/causal*conv1d*cp312*torch2.10*.whl 2>/dev/null | tail -1)
MAMBA=$(ls "$PKG1"/mamba_ssm-*cp312*torch2.10*.whl 2>/dev/null | tail -1)
log "  causal=$CAUSAL"; log "  mamba=$MAMBA"
[ -n "$CAUSAL" ] && uv pip install --python "$PY" --no-index --no-deps "$CAUSAL" 2>&1 | tail -2
[ -n "$MAMBA" ]  && uv pip install --python "$PY" --no-index --no-deps "$MAMBA"  2>&1 | tail -2

log "4) vllm 0.18.0 (for local eval harness)"
uv pip install --python "$PY" --no-index $FL "vllm==0.18.0" 2>&1 | tail -5

log "5) freeze -> foundation/requirements.lock"
uv pip freeze --python "$PY" > foundation/requirements.lock 2>/dev/null
log "lock lines: $(wc -l < foundation/requirements.lock)"

log "6) import smoke test"
"$PY" - <<'PYEOF' 2>&1 | tail -20
import sys
print("python", sys.version.split()[0])
mods = ["torch","transformers","peft","trl","datasets","accelerate","mamba_ssm","causal_conv1d","triton","vllm"]
import importlib
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"OK  {m} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"ERR {m}: {type(e).__name__}: {str(e)[:120]}")
import torch
print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PYEOF
log "BUILD_VENV DONE"
