from __future__ import annotations

import argparse
import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

from config import DEFAULT_LOCOMO_PATH, PROJECT_ROOT


REMOTE_ROOT = Path("/workspace")
REMOTE_RESULTS = Path("/tmp/kdd-modal-results")


def run_real_on_modal(args: argparse.Namespace) -> dict[str, Path]:
    if args.api_key:
        raise SystemExit("Modal runs require env-var secrets; do not pass --api-key.")
    modal = _import_modal()
    if args.modal_call_id:
        result = modal.FunctionCall.from_id(args.modal_call_id).get(timeout=args.modal_timeout)
        return _handle_result(result, args.out_dir)
    manifest = _dataset_manifest(args)
    command = _remote_command(args)
    app = modal.App(args.modal_app_name)
    image = _modal_image(modal)
    secrets = _modal_secrets(modal, args)
    function_kwargs: dict[str, Any] = {"image": image, "timeout": args.modal_timeout, "serialized": True}
    if args.modal_cpu:
        function_kwargs["cpu"] = args.modal_cpu
    if args.modal_memory:
        function_kwargs["memory"] = args.modal_memory
    if args.modal_gpu:
        function_kwargs["gpu"] = args.modal_gpu
    if secrets:
        function_kwargs["secrets"] = secrets

    @app.function(**function_kwargs)
    def _run_real(command: list[str], manifest: dict[str, Any]) -> dict[str, Any]:
        import subprocess
        import zipfile
        from io import BytesIO
        from pathlib import Path

        out_dir = Path("/tmp/kdd-modal-results")
        out_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(command, cwd="/workspace", text=True, capture_output=True)
        files = []
        if out_dir.exists():
            files = [path for path in out_dir.rglob("*") if path.is_file()]
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in files:
                zf.write(path, path.relative_to(out_dir).as_posix())
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "manifest": manifest,
            "artifact_zip": buffer.getvalue(),
            "files": [path.relative_to(out_dir).as_posix() for path in files],
        }

    with app.run(detach=args.modal_detach):
        if args.modal_detach:
            call = _run_real.spawn(command, manifest)
            print(f"Modal call id: {call.object_id}")
            print("Fetch artifacts later with:")
            print(f"  python experiment/run.py --runner modal --mode real --modal-call-id {call.object_id}")
            return {}
        result = _run_real.remote(command, manifest)
    return _handle_result(result, args.out_dir)


def _import_modal():
    try:
        import modal
    except Exception as exc:
        raise SystemExit("Modal is not installed. Install it with `pip install modal` and authenticate with `modal setup`.") from exc
    return modal


def _modal_image(modal):
    image = modal.Image.debian_slim(python_version="3.12").pip_install("matplotlib", "rhesis-sdk")
    image = image.add_local_dir(str(PROJECT_ROOT / "experiment"), remote_path="/workspace/experiment")
    image = image.add_local_dir(str(PROJECT_ROOT / "memory-probe" / "data"), remote_path="/workspace/memory-probe/data")
    real_data = PROJECT_ROOT / "experiment" / "data" / "real"
    if real_data.exists():
        image = image.add_local_dir(str(real_data), remote_path="/workspace/experiment/data/real")
    return image


def _modal_secrets(modal, args: argparse.Namespace):
    values = {}
    for env_name in {args.api_key_env, args.rhesis_api_key_env, "RHESIS_BASE_URL"}:
        value = os.environ.get(env_name)
        if value:
            values[env_name] = value
    return [modal.Secret.from_dict(values)] if values else []


def _dataset_manifest(args: argparse.Namespace) -> dict[str, Any]:
    datasets = set(args.datasets)
    manifest: dict[str, Any] = {"datasets": sorted(datasets), "files": []}
    if "locomo" in datasets:
        _require_file(args.locomo_path)
        manifest["files"].append(_file_record("locomo", args.locomo_path))
    if "longmemeval" in datasets:
        for file_name in args.longmemeval_files:
            path = args.longmemeval_dir / file_name
            _require_file(path)
            manifest["files"].append(_file_record("longmemeval", path))
    if "memoryarena" in datasets:
        files = sorted(args.memoryarena_dir.glob("*.jsonl"))
        if not files:
            raise SystemExit(f"No MemoryArena JSONL files found in {args.memoryarena_dir}")
        for path in files:
            manifest["files"].append(_file_record("memoryarena", path))
    return manifest


def _require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Required dataset file not found: {path}")


def _file_record(dataset: str, path: Path) -> dict[str, Any]:
    return {"dataset": dataset, "name": path.name, "bytes": path.stat().st_size}


def _remote_command(args: argparse.Namespace) -> list[str]:
    command = [
        "python",
        "/workspace/experiment/run.py",
        "--mode", "real",
        "--runner", "local",
        "--backend", args.backend,
        "--strategies", *args.strategies,
        "--seed", str(args.seed),
        "--top-k", str(args.top_k),
        "--target-score", str(args.target_score),
        "--out-dir", str(REMOTE_RESULTS),
        "--locomo-path", str(REMOTE_ROOT / "memory-probe" / "data" / DEFAULT_LOCOMO_PATH.name),
        "--max-conversations", str(args.max_conversations),
        "--max-questions", str(args.max_questions),
        "--datasets", *args.datasets,
        "--longmemeval-dir", str(REMOTE_ROOT / "experiment" / "data" / "real" / "longmemeval"),
        "--longmemeval-files", *args.longmemeval_files,
        "--memoryarena-dir", str(REMOTE_ROOT / "experiment" / "data" / "real" / "memoryarena"),
        "--base-url", args.base_url,
        "--model", args.model,
        "--api-key-env", args.api_key_env,
        "--eval-backend", args.eval_backend,
        "--rhesis-api-key-env", args.rhesis_api_key_env,
        "--semantica-mode", args.semantica_mode,
    ]
    if args.max_items is not None:
        command.extend(["--max-items", str(args.max_items)])
    if args.eval_limit is not None:
        command.extend(["--eval-limit", str(args.eval_limit)])
    if args.eval_smoke_test:
        command.append("--eval-smoke-test")
    if not args.eval_fail_open:
        command.append("--no-eval-fail-open")
    if args.rhesis_base_url:
        command.extend(["--rhesis-base-url", args.rhesis_base_url])
    if args.rhesis_model:
        command.extend(["--rhesis-model", args.rhesis_model])
    if args.visualize and args.modal_include_figures:
        command.append("--visualize")
    return command


def _handle_result(result: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    if result["stdout"]:
        print(result["stdout"], end="")
    if result["stderr"]:
        print(result["stderr"], end="")
    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])
    return _unpack_artifacts(result["artifact_zip"], out_dir)


def _unpack_artifacts(payload: bytes, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(payload), "r") as zf:
        zf.extractall(out_dir)
    paths = {
        "raw": _latest(out_dir, "run_*_real_raw.json"),
        "metrics": _latest(out_dir, "run_*_real_metrics.json"),
        "summary": _latest(out_dir, "run_*_real_summary.tsv"),
    }
    print("\nModal artifacts synced:")
    for label, path in paths.items():
        if path:
            print(f"  {label}: {path}")
    return {key: path for key, path in paths.items() if path}


def _latest(out_dir: Path, pattern: str) -> Path | None:
    paths = sorted(out_dir.glob(pattern))
    return paths[-1] if paths else None
