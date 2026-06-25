#!/usr/bin/env python3
"""Prepare Workspace-Bench data for slime RL and stage it to S3.

Pipeline (run on a dev box or the Greenland head node):
  1. Download the task dataset (per-task metadata.json [+ data/ manifest files])
     from Hugging Face for the chosen split (lite | full).
  2. Download + unzip the large persona workspaces (filesys_en_workdirs.zip),
     renaming the extracted dirs to the *_raw names the env expects.
  3. Normalize each task's `file_system` to the *_raw workspace dir name and copy
     each task folder into <out>/tasks/<task_id>/.
  4. Write slime prompt-data: ws_<split>_train_tasks.jsonl / ws_<split>_dev_tasks.jsonl
     (rows {"index": i}) + task_index_map.json (index -> task folder, per split).
  5. Upload <out>/ to s3://whx-agent/data/workspace-bench/ (profile greenland-dev).

At submit time, `--stage-data workspace-bench/` makes the Greenland bootstrap
`aws s3 sync` this back to $DATA_ROOT/workspace-bench/, where WorkspaceEnv reads it.

Usage:
  python3 prepare_ws_data.py --split full
  python3 prepare_ws_data.py --split lite --skip-upload      # local dry-run
  python3 prepare_ws_data.py --split full --skip-download     # re-stage existing
"""

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile

# HF dataset repos (from Workspace-Bench evaluation/scripts/download_hf_assets.py).
DATASETS = {
    "lite": "Workspace-Bench/Workspace-Bench-Lite",
    "full": "Workspace-Bench/Workspace-Bench",
}
WORKSPACES_REPO = "Workspace-Bench/Workspace-Bench-Workspaces"
# The workspaces repo ships per-language archives: filesys_en.zip (18.7 GB) /
# filesys_cn.zip. (The upstream download_hf_assets.py constant
# "filesys_en_workdirs.zip" does NOT exist in the repo — 404.)
WORKSPACE_ARCHIVE_FMT = "filesys_{lang}.zip"

# Extracted-zip dir -> the *_raw name the env mounts (verbatim from upstream).
WORKSPACE_EXTRACTED_DIR_MAP = {
    "ProductManager_Workdir": "chanpin_raw",
    "BackendDeveloper_Workdir": "kaifa_raw",
    "Research_Workdir": "research_raw",
    "OperationsManager_Workdir": "yunying_raw",
    "LogisticsManager_Workdir": "houqin_raw",
}

# Normalize a task's `file_system` (which upstream sets to a localized label) OR
# its English persona to the *_raw workspace dir. Belt-and-suspenders: base.py
# carries the same maps as a runtime fallback.
PERSONA_TO_WORKDIR = {
    "Product Manager": "chanpin_raw",
    "Backend Developer": "kaifa_raw",
    "Researcher": "research_raw",
    "Operations Manager": "yunying_raw",
    "Logistics Manager": "houqin_raw",
}
FILE_SYSTEM_LABEL_TO_WORKDIR = {
    "产品人员": "chanpin_raw",
    "开发人员": "kaifa_raw",
    "研究人员": "research_raw",
    "运营人员": "yunying_raw",
    "行政/后勤人员": "houqin_raw",
}
S3_PREFIX_DEFAULT = "s3://whx-agent/data/workspace-bench/"


def _ensure_hf():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print(">> installing huggingface_hub + hf_transfer ...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--user", "--quiet", "huggingface_hub", "hf_transfer"],
            check=True,
        )
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _resolve_workdir(meta) -> str:
    """Map a task's metadata to its *_raw workspace dir name (or '' if none)."""
    fs = meta.get("file_system")
    if fs:
        if fs in WORKSPACE_EXTRACTED_DIR_MAP.values():  # already a *_raw name
            return fs
        if fs in FILE_SYSTEM_LABEL_TO_WORKDIR:
            return FILE_SYSTEM_LABEL_TO_WORKDIR[fs]
        if fs in WORKSPACE_EXTRACTED_DIR_MAP:  # an extracted-dir name
            return WORKSPACE_EXTRACTED_DIR_MAP[fs]
    persona = meta.get("persona")
    if persona and persona in PERSONA_TO_WORKDIR:
        return PERSONA_TO_WORKDIR[persona]
    return fs or ""


def _safe_task_id(meta, fallback_idx) -> str:
    tid = meta.get("id") or meta.get("absolute_id")
    if tid is not None:
        return str(tid)
    persona = (meta.get("persona") or "task").lower().replace(" ", "_").replace("/", "_")
    return f"{persona}_{fallback_idx}"


def download_tasks(split: str, work: str, lang: str = "en") -> str:
    from huggingface_hub import snapshot_download

    repo = DATASETS[split]
    dst = os.path.join(work, f"hf_tasks_{split}")
    tok = os.environ.get("HF_TOKEN") or None
    # The repo ships BOTH task_clean_en/ and task_clean_cn/ (same ids); pull only
    # the requested language so we don't 2x the download nor collide en/cn at the
    # same tasks/<id>/. Lite (Workspace-Bench-Lite) may not be language-split, so
    # fall back to a full snapshot if the lang prefix matches nothing.
    # NOTE: only the language tree — NOT bare "*.json"/"*.csv". fnmatch '*' crosses
    # '/', so "*.json" would also pull task_clean_cn/.../metadata.json and defeat
    # the filter. The task_clean_<lang>/ tree already contains every per-task
    # metadata.json + data/ file stage_tasks needs.
    print(f">> downloading task dataset {repo} (lang={lang}) -> {dst}", flush=True)
    try:
        snapshot_download(
            repo_id=repo, repo_type="dataset", local_dir=dst, local_dir_use_symlinks=False,
            token=tok, allow_patterns=[f"task_clean_{lang}/*"],
        )
    except Exception as e:
        print(f"   (lang-filtered download failed [{type(e).__name__}: {e}]; full snapshot)", flush=True)
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dst,
                          local_dir_use_symlinks=False, token=tok)
    return dst


def download_workspaces(work: str, lang: str = "en") -> str:
    from huggingface_hub import hf_hub_download

    tok = os.environ.get("HF_TOKEN") or None
    fname = WORKSPACE_ARCHIVE_FMT.format(lang=lang)
    print(f">> downloading workspaces archive {fname} from {WORKSPACES_REPO}", flush=True)
    archive = hf_hub_download(
        repo_id=WORKSPACES_REPO, filename=fname, repo_type="dataset",
        local_dir=os.path.join(work, "hf_workspaces"), local_dir_use_symlinks=False, token=tok,
    )
    extract_root = os.path.join(work, "ws_extract")
    os.makedirs(extract_root, exist_ok=True)
    print(f">> unzipping {archive} -> {extract_root}", flush=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract_root)
    return extract_root


def stage_workspaces(extract_root: str, out: str):
    ws_out = os.path.join(out, "workspaces")
    os.makedirs(ws_out, exist_ok=True)
    # Find each extracted persona workdir (possibly nested) and rename to *_raw.
    for src_name, raw_name in WORKSPACE_EXTRACTED_DIR_MAP.items():
        matches = glob.glob(os.path.join(extract_root, "**", src_name), recursive=True)
        if not matches:
            print(f"   (workspace dir {src_name} not found in archive — skipping)", flush=True)
            continue
        dst = os.path.join(ws_out, raw_name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(matches[0], dst)
        print(f"   staged workspace {src_name} -> workspaces/{raw_name}", flush=True)


def stage_tasks(hf_tasks_dir: str, out: str, lang: str = "en") -> list:
    """Copy each per-task folder into out/tasks/<task_id>/, normalizing file_system.
    Returns the ordered list of task_ids."""
    tasks_out = os.path.join(out, "tasks")
    os.makedirs(tasks_out, exist_ok=True)
    # The repo nests tasks under task_clean_<lang>/<id>/metadata.json AND ships a
    # second-language tree with the SAME ids. Anchor to ONE language dir so we
    # never mix en/cn into the same tasks/<id>/. Fall back to a recursive scan
    # only if the language dir is absent (e.g. the Lite repo isn't lang-split).
    lang_root = os.path.join(hf_tasks_dir, f"task_clean_{lang}")
    scan_root = lang_root if os.path.isdir(lang_root) else hf_tasks_dir
    meta_paths = sorted(glob.glob(os.path.join(scan_root, "**", "metadata.json"), recursive=True))
    if not meta_paths:
        raise SystemExit(
            f"No metadata.json found under {scan_root}. The HF dataset layout may "
            f"differ (e.g. a CSV/parquet); inspect it and extend stage_tasks()."
        )
    print(f">> scanning tasks under {scan_root} ({len(meta_paths)} metadata.json)", flush=True)
    task_ids = []
    for idx, mp in enumerate(meta_paths):
        with open(mp) as f:
            meta = json.load(f)
        task_id = _safe_task_id(meta, idx)
        meta["file_system"] = _resolve_workdir(meta)  # normalize to *_raw
        src_dir = os.path.dirname(mp)
        dst_dir = os.path.join(tasks_out, task_id)
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        # Copy the whole task folder (metadata.json + any data/ manifest files),
        # then overwrite metadata.json with the normalized version.
        shutil.copytree(src_dir, dst_dir)
        with open(os.path.join(dst_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False)
        task_ids.append(task_id)
    print(f">> staged {len(task_ids)} tasks -> {tasks_out}", flush=True)
    return task_ids


def write_prompt_data(task_ids: list, out: str, split: str, train_frac: float, seed: int):
    ids = list(task_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_dev = max(1, int(round(len(ids) * (1.0 - train_frac)))) if len(ids) > 1 else 0
    dev_ids = ids[:n_dev]
    train_ids = ids[n_dev:]

    index_map = {"train": train_ids, "dev": dev_ids}
    with open(os.path.join(out, "task_index_map.json"), "w") as f:
        json.dump(index_map, f, ensure_ascii=False)

    for sp, lst in (("train", train_ids), ("dev", dev_ids)):
        path = os.path.join(out, f"ws_{split}_{sp}_tasks.jsonl")
        with open(path, "w") as f:
            for i in range(len(lst)):
                f.write(json.dumps({"index": i}) + "\n")
        print(f">> wrote {path} ({len(lst)} tasks)", flush=True)


def upload_s3(out: str, s3_prefix: str, profile: str):
    cmd = ["aws", "s3", "sync", out.rstrip("/") + "/", s3_prefix, "--no-progress"]
    if profile:
        cmd += ["--profile", profile]
    print(">> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    subprocess.run(["aws", "s3", "ls", s3_prefix] + (["--profile", profile] if profile else []), check=False)


def main():
    ap = argparse.ArgumentParser(description="Prepare + stage Workspace-Bench data for slime RL.")
    ap.add_argument("--split", choices=["lite", "full"], default="full")
    ap.add_argument("--lang", choices=["en", "cn"], default="en",
                    help="task language tree to stage (repo ships both task_clean_en/ and task_clean_cn/).")
    ap.add_argument("--work", default="/local/home/whx/ws-bench-downloads", help="scratch dir for HF downloads")
    ap.add_argument("--out", default=None, help="staging dir mirroring WS_DATA_DIR (default <work>/staged_<split>)")
    ap.add_argument("--s3-prefix", default=S3_PREFIX_DEFAULT)
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE_NAME", "greenland-dev"))
    ap.add_argument("--train-frac", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--skip-download", action="store_true", help="reuse already-downloaded HF data under --work")
    ap.add_argument("--skip-workspaces", action="store_true", help="skip the large workspaces archive")
    ap.add_argument("--skip-upload", action="store_true", help="local-only; do not sync to S3")
    args = ap.parse_args()

    out = args.out or os.path.join(args.work, f"staged_{args.split}")
    os.makedirs(out, exist_ok=True)
    _ensure_hf()

    if args.skip_download:
        hf_tasks_dir = os.path.join(args.work, f"hf_tasks_{args.split}")
    else:
        hf_tasks_dir = download_tasks(args.split, args.work, lang=args.lang)
        if not args.skip_workspaces:
            extract_root = download_workspaces(args.work, lang=args.lang)
            stage_workspaces(extract_root, out)

    task_ids = stage_tasks(hf_tasks_dir, out, lang=args.lang)
    write_prompt_data(task_ids, out, args.split, args.train_frac, args.seed)

    # Quick footprint report (Full workspaces can be ~20 GB — measure before trust).
    du = subprocess.run(["du", "-sh", out], capture_output=True, text=True)
    print(f">> staged size: {du.stdout.strip()}", flush=True)

    if args.skip_upload:
        print(">> --skip-upload set; staged locally at", out, flush=True)
    else:
        upload_s3(out, args.s3_prefix, args.profile)
    print(">> DONE", flush=True)


if __name__ == "__main__":
    main()
