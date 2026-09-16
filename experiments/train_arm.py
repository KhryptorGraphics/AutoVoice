"""Train one arm of the keyshift-augmentation programme and print its metrics.

Every node runs this same file with the same command; only config.json differs.
The metrics printed at the end are the contract with the experiment tree:

  DROPOUT_700  fraction of source-voiced frames above 700 Hz where the converted
               vocal sits >8 dB below its source-relative norm. This is the
               defect the user hears - 59 % on the served checkpoint.
  IDENTITY     speaker-centroid similarity against her own corpus. FLOOR 0.88.
               Resemblyzer is NOT usable here (it scores two different singers
               0.899 vs 0.907); this is the in-house mel-statistics centroid.
  POST_ERR_700 posterior reconstruction error above 700 Hz, which separates
               "decoder undertrained in this register" from "inference failing".

An arm that improves DROPOUT_700 while pushing IDENTITY below the floor has not
won. Both numbers decide it, and the tree keeps whichever arm moves the first
without breaking the second.
"""
import json, os, subprocess, sys, time

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main(cfg_path):
    cfg = json.load(open(cfg_path))
    arm = cfg["arm"]; shifts = cfg.get("shifts", [])
    log(f"ARM={arm} shifts={shifts} epochs={cfg['epochs']} batch={cfg['batch_size']}")

    root = os.environ.get("ARM_DATA_ROOT", "/root/arms")
    data = os.path.join(root, arm)
    if not os.path.isdir(data):
        log(f"FATAL: corpus for this arm is missing at {data}")
        log("Build it with experiments/build_corpus.py and sync it to the box.")
        return 1
    n = len([f for f in os.listdir(os.path.join(data, "dataset_raw", "brandy"))
             if f.endswith(".wav")]) if os.path.isdir(os.path.join(data, "dataset_raw", "brandy")) else 0
    tagged = len([f for f in os.listdir(os.path.join(data, "dataset_raw", "brandy"))
                  if "_ks" in f]) if n else 0
    log(f"corpus: {n} items, {tagged} keyshift-tagged ({tagged/max(n,1):.0%})")

    # Preprocessing. -n 2 is deliberate: the default worker count OOM'd a
    # previous box during hubert extraction and silently dropped 65 of 438 files.
    # This box has 15 GB of RAM, so it is tighter still.
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    for step in (["svc", "pre-resample"], ["svc", "pre-config"],
                 ["svc", "pre-hubert", "-n", "2"]):
        log(f"$ {' '.join(step)}")
        r = subprocess.run(step, cwd=data, env=env)
        if r.returncode != 0:
            log(f"FATAL: {' '.join(step)} exited {r.returncode}")
            return r.returncode

    log("$ svc train")
    r = subprocess.run(["svc", "train", "-t"], cwd=data, env=env)
    log(f"train exited {r.returncode}")
    log("RESULT_PENDING: evaluate checkpoints with experiments/score_arm.py")
    return r.returncode

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "experiments/config.json"))
