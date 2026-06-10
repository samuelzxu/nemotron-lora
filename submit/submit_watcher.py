"""Auto-submit grid adapters to the LB the moment each epoch checkpoint lands.

Watches artifacts/grid_*/adapter_ep{1,2}/.done (written by train/train_sft.py's
SaveEpochAdapter callback). For each newly-completed adapter, runs the PROVEN
code-competition pipeline serially:

  1. stage adapter + model-instance-metadata.json   -> kaggle_upload/<slug>/
  2. kaggle models instances create                 (uploads ~3.5GB)
  3. poll `instances versions files .../<slug>/1`    until adapter_config.json registers
  4. rewrite kaggle_submission/kernel-metadata.json  model_sources -> the new instance
  5. kaggle kernels push                             (no-accel kernel zips adapter -> submission.zip)
  6. poll `kernels status`                           until COMPLETE
  7. kaggle competitions submit -k <kernel> -f submission.zip -m <desc>

State in submit/state.json. Serial by construction (shared kernel + metadata file).
Daily-quota / transient errors -> status 'retry', re-attempted next cycle.

Run (creds inherited from a sourced ~/.claudeinit; never printed):
  source ~/.claudeinit && nohup foundation/.venv/bin/python submit/submit_watcher.py \
      > logs/submit_watcher.log 2>&1 &
"""
import json, os, re, subprocess, sys, time, datetime

ROOT = "/home/ec2-user/dev/nemotron_training"
os.chdir(ROOT)
COMP = "nvidia-nemotron-model-reasoning-challenge"
MODEL = "samuelzxu/nemotron-nano-30b-trained"
FRAMEWORK = "Transformers"            # path casing that the working kernel uses
KERNEL = "samuelzxu/nemotron-submission"
KERNEL_DIR = "kaggle_submission"
KERNEL_META = f"{KERNEL_DIR}/kernel-metadata.json"
STAGE_ROOT = "kaggle_upload"
STATE = "submit/state.json"

RUNS = ["eqguess_3x_a64lr1e4_3ep", "full_eqguess_3x_a64lr1e4_3ep"]  # finish 3x ep3 + full-dataset run
EPOCHS = [1, 2, 3]
POLL_CYCLE = 300                      # seconds between scan cycles
MODEL_READY_TRIES, MODEL_READY_WAIT = 80, 30     # up to ~40 min
KERNEL_DONE_TRIES, KERNEL_DONE_WAIT = 90, 20      # up to ~30 min


def now():
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def run(cmd, timeout=1800):
    """Run a command, return (rc, combined_output). Never raises."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"
    except Exception as e:
        return 1, f"{type(e).__name__}: {e}"


def load_state():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE))
        except Exception:
            pass
    return {}


def save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w"), indent=2)
    os.replace(tmp, STATE)


def slug_for(run_name, ep):
    # grid_a32_lr1e-4 -> grid-a32-lr1e-4-ep1  (lowercase, hyphens; valid kaggle slug)
    return f"{run_name.replace('_', '-')}-ep{ep}"


def desc_for(run_name, ep):
    return (f"{run_name} epoch{ep}/3 (best hp: a64/lr1e-4/cosine/wu0.03, rank32) "
            f"+ eqguess distill (captures+solver)")


def stage_adapter(adapter_dir, slug, desc):
    stage = f"{STAGE_ROOT}/{slug}"
    os.makedirs(stage, exist_ok=True)
    for fn in ("adapter_config.json", "adapter_model.safetensors"):
        src = os.path.join(adapter_dir, fn)
        if not os.path.exists(src):
            return None, f"missing {fn}"
        # hardlink if possible (same fs) else copy
        dst = os.path.join(stage, fn)
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(src, dst)
        except OSError:
            import shutil
            shutil.copy2(src, dst)
    meta = {
        "ownerSlug": "samuelzxu",
        "modelSlug": "nemotron-nano-30b-trained",
        "instanceSlug": slug,
        "framework": "transformers",
        "overview": desc,
        "usage": "Mount + zip adapter_config.json + adapter_model.safetensors into submission.zip.",
        "licenseName": "Apache 2.0",
        "fineTunable": False,
    }
    json.dump(meta, open(f"{stage}/model-instance-metadata.json", "w"), indent=2)
    return stage, None


def model_ready(slug):
    path = f"{MODEL}/{FRAMEWORK}/{slug}/1"
    rc, out = run(["kaggle", "models", "instances", "versions", "files", path], timeout=120)
    return "adapter_config.json" in out


def point_kernel_at(slug):
    meta = json.load(open(KERNEL_META))
    meta["model_sources"] = [f"{MODEL}/{FRAMEWORK}/{slug}/1"]
    json.dump(meta, open(KERNEL_META, "w"), indent=2)


def kernel_status():
    rc, out = run(["kaggle", "kernels", "status", KERNEL], timeout=120)
    return out


def submit_one(adapter_dir, run_name, ep, st):
    slug = slug_for(run_name, ep)
    desc = desc_for(run_name, ep)
    rec = st.setdefault(slug, {"run": run_name, "ep": ep, "status": "new"})
    log(f"--- submitting {slug} ({adapter_dir}) ---")

    # 1-2. stage + create instance (idempotent-ish: tolerate 'already exists')
    stage, err = stage_adapter(adapter_dir, slug, desc)
    if err:
        rec["status"] = "retry"; rec["err"] = err; return
    rc, out = run(["kaggle", "models", "instances", "create", "-p", stage], timeout=3600)
    if rc != 0 and "exist" not in out.lower():
        log(f"{slug}: instance create rc={rc}: {out[-300:]}")
        rec["status"] = "retry"; rec["err"] = out[-300:]; return

    # 3. wait until the model version registers
    ready = False
    for _ in range(MODEL_READY_TRIES):
        if model_ready(slug):
            ready = True; break
        time.sleep(MODEL_READY_WAIT)
    if not ready:
        log(f"{slug}: model never became ready"); rec["status"] = "retry"; return
    log(f"{slug}: model ready")

    # 4-5. point kernel + push (capture the new version number)
    point_kernel_at(slug)
    rc, out = run(["kaggle", "kernels", "push", "-p", KERNEL_DIR], timeout=600)
    if rc != 0:
        log(f"{slug}: kernel push rc={rc}: {out[-300:]}"); rec["status"] = "retry"; return
    vm = re.search(r"version (\d+)", out)
    if not vm:
        log(f"{slug}: no version in push output: {out[-200:]}"); rec["status"] = "retry"; return
    version = vm.group(1)
    log(f"{slug}: pushed kernel v{version}")

    # 6-7. CODE submission. This is a code competition: plain-file submit (CreateSubmission) is
    #      rejected with 400; only `-k -v` (CreateCodeSubmission) is valid, and it needs a COMPLETED
    #      kernel run. The kernels-status endpoint is unreliable (HTTP 500), so instead of polling it
    #      we retry the code-submit (it errors while the kernel runs, is accepted once it finishes)
    #      and confirm via the submissions board. Idempotent: skip if already on the board.
    marker = f"{run_name} epoch{ep}/3"

    def on_board():
        _, o = run(["kaggle", "competitions", "submissions", "-c", COMP], timeout=120)
        return marker in o

    time.sleep(240)                      # kernel runs ~4-5 min
    submitted = on_board()
    for _ in range(15):
        if submitted:
            break
        rc, out = run(["kaggle", "competitions", "submit", "-c", COMP, "-k", KERNEL,
                       "-v", version, "-f", "submission.zip", "-m", desc], timeout=300)
        if any(k in out.lower() for k in ("exceeded", "limit", "maximum number", "too many")):
            log(f"{slug}: QUOTA hit, will retry next cycle"); rec["status"] = "retry"; rec["err"] = "quota"; return
        time.sleep(20)
        submitted = on_board()
        if not submitted:
            time.sleep(60)
    if not submitted:
        log(f"{slug}: code-submit not accepted yet; will retry next cycle"); rec["status"] = "retry"; return
    rec["status"] = "submitted"; rec["submitted_at"] = now(); rec.pop("err", None)
    log(f"{slug}: SUBMITTED (code v{version})")


def main():
    log(f"submit_watcher start; expecting {len(RUNS)*len(EPOCHS)} adapters")
    while True:
        st = load_state()
        pending = []
        for run_name in RUNS:
            for ep in EPOCHS:
                slug = slug_for(run_name, ep)
                if st.get(slug, {}).get("status") == "submitted":
                    continue
                done_marker = f"artifacts/{run_name}/adapter_ep{ep}/.done"
                if os.path.exists(done_marker):
                    pending.append((f"artifacts/{run_name}/adapter_ep{ep}", run_name, ep))
        for adapter_dir, run_name, ep in pending:
            try:
                submit_one(adapter_dir, run_name, ep, st)
            except Exception as e:
                log(f"submit_one({run_name} ep{ep}) crashed: {type(e).__name__}: {e}")
                st.setdefault(slug_for(run_name, ep), {"run": run_name, "ep": ep})["status"] = "retry"
            save_state(st)

        st = load_state()
        expected = {slug_for(r, e) for r in RUNS for e in EPOCHS}
        n_sub = sum(1 for s in expected if st.get(s, {}).get("status") == "submitted")
        if n_sub >= len(expected):
            log(f"all {n_sub}/{len(expected)} target adapters submitted; watcher exiting")
            return
        time.sleep(POLL_CYCLE)


if __name__ == "__main__":
    main()
