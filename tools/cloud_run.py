#!/usr/bin/env python3
"""Resumable training wrapper for Open Duck Playground (Kaggle / Colab / any Linux).

Env vars:
  HF_REPO       private Hub repo for state, e.g. "youruser/open-duck-ckpt"  (required)
  HF_TOKEN      Hugging Face WRITE token (used automatically by huggingface_hub)
  RUN_TAG       name of this experiment (default: flat_terrain_v1)
  TASK          runner --task value (default: flat_terrain)
  TARGET_STEPS  total env steps wanted across all sessions (default: 150000000)
  UPLOAD_EVERY_S  seconds between upload scans (default: 300)
"""
import json
import os
import shutil
import subprocess
import tarfile
import threading
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

try:
    from huggingface_hub.errors import EntryNotFoundError
except ImportError:  # older huggingface_hub
    from huggingface_hub.utils import EntryNotFoundError

HF_REPO = os.environ["HF_REPO"]
RUN_TAG = os.environ.get("RUN_TAG", "flat_terrain_v1")
TASK = os.environ.get("TASK", "flat_terrain")
TARGET = int(os.environ.get("TARGET_STEPS", "150000000"))
UPLOAD_EVERY = int(os.environ.get("UPLOAD_EVERY_S", "300"))
MIN_REMAINING = 3_000_000  # below this Brax may not complete one update per eval
SETTLE_S = int(os.environ.get("ONNX_SETTLE_S", "20"))  # wait until the .onnx file stops changing

api = HfApi()
work = Path("cloud_work")
work.mkdir(exist_ok=True)
MANIFEST_REMOTE = f"{RUN_TAG}/manifest.json"
MANIFEST_LOCAL = work / "manifest.json"


def load_manifest():
    api.create_repo(HF_REPO, private=True, exist_ok=True)
    try:
        p = hf_hub_download(HF_REPO, MANIFEST_REMOTE, local_dir=str(work / "remote"), force_download=True)
    except EntryNotFoundError:
        print("No manifest yet: first session of this run.")
        return {"target_total": TARGET, "segments": [], "checkpoints": []}
    return json.loads(Path(p).read_text())  # any other error stops the script on purpose


def push_manifest(m):
    MANIFEST_LOCAL.write_text(json.dumps(m, indent=2))
    api.upload_file(path_or_fileobj=str(MANIFEST_LOCAL), path_in_repo=MANIFEST_REMOTE, repo_id=HF_REPO)


class Uploader(threading.Thread):
    def __init__(self, outdir, seg, base, manifest):
        super().__init__(daemon=True)
        self.outdir, self.seg, self.base, self.m = Path(outdir), seg, base, manifest
        self.seen = {c["name"] for c in manifest["checkpoints"] if c["seg"] == seg}
        self.stop_ev = threading.Event()
        self.event_sizes = {}

    def upload_events(self):
        for ev in self.outdir.glob("events.out.tfevents.*"):
            size = ev.stat().st_size
            if self.event_sizes.get(ev.name) == size:
                continue
            api.upload_file(path_or_fileobj=str(ev),
                            path_in_repo=f"{RUN_TAG}/tb/seg{self.seg:03d}/{ev.name}", repo_id=HF_REPO)
            self.event_sizes[ev.name] = size

    def scan_once(self, final=False):
        self.upload_events()
        # a checkpoint dir is complete once its .onnx exists (runner exports ONNX after saving)
        for onnx in sorted(self.outdir.glob("*.onnx")):
            name = onnx.stem
            ckdir = self.outdir / name
            if name in self.seen or not ckdir.is_dir():
                continue
            if not final and time.time() - onnx.stat().st_mtime < SETTLE_S:
                continue  # file may still be written
            step = int(name.rsplit("_", 1)[1])
            tar_local = self.outdir / f"{name}.tar"
            with tarfile.open(tar_local, "w") as t:
                t.add(ckdir, arcname=name)
            tar_remote = f"{RUN_TAG}/ckpt/seg{self.seg:03d}/{name}.tar"
            onnx_remote = f"{RUN_TAG}/onnx/seg{self.seg:03d}/{name}.onnx"
            api.upload_file(path_or_fileobj=str(tar_local), path_in_repo=tar_remote, repo_id=HF_REPO)
            api.upload_file(path_or_fileobj=str(onnx), path_in_repo=onnx_remote, repo_id=HF_REPO)
            self.m["checkpoints"].append({
                "seg": self.seg, "name": name, "step": step, "cum": self.base + step,
                "tar": tar_remote, "onnx": onnx_remote,
            })
            push_manifest(self.m)
            self.seen.add(name)
            tar_local.unlink()
            print(f"[uploader] saved {name} (cumulative steps {self.base + step:,})", flush=True)

    def run(self):
        while not self.stop_ev.wait(UPLOAD_EVERY):
            try:
                self.scan_once()
            except Exception as e:  # network hiccup: retry next round
                print("[uploader] error, will retry:", repr(e), flush=True)


def main():
    m = load_manifest()
    m["target_total"] = TARGET
    best = max(m["checkpoints"], key=lambda c: c["cum"], default=None)
    done = best["cum"] if best else 0
    remaining = TARGET - done
    print(f"Done so far: {done:,} / {TARGET:,} steps. Remaining: {remaining:,}", flush=True)
    if remaining < MIN_REMAINING:
        print("Target reached (or remaining too small). Nothing to train.")
        return 0

    seg = len(m["segments"]) + 1
    m["segments"].append({"index": seg, "base": done, "task": TASK,
                          "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    push_manifest(m)

    outdir = Path("segments") / f"seg_{seg:03d}"
    cmd = ["uv", "run", "playground/open_duck_mini_v2/runner.py",
           "--task", TASK, "--num_timesteps", str(remaining), "--output_dir", str(outdir)]
    if best:
        tar_path = hf_hub_download(HF_REPO, best["tar"], local_dir=str(work / "remote"))
        dest = work / "restore"
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        with tarfile.open(tar_path) as t:
            t.extractall(dest)
        cmd += ["--restore_checkpoint_path", str(dest / best["name"])]

    up = Uploader(outdir, seg, done, m)
    up.start()
    print("Running:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    up.stop_ev.set()
    up.join(timeout=1)
    up.scan_once(final=True)
    print("Runner exit code:", rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())