"""Run a selected catalog batch with bounded process concurrency."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


from spatialscout.paths import ROOT, CATALOG_PATH


def active_batch_pids(output_root: Path) -> list[int]:
    """Find older batch runners targeting this directory (including pre-lock versions)."""
    active = []
    for process in Path("/proc").glob("[0-9]*"):
        pid = int(process.name)
        if pid == os.getpid():
            continue
        try:
            args = (process / "cmdline").read_bytes().split(b"\0")
            args = [arg.decode(errors="replace") for arg in args if arg]
        except (OSError, PermissionError):
            continue
        if not any(arg == "spatialscout.benchmark.batch" or Path(arg).name == "run_catalog_batch.py" for arg in args):
            continue
        for index, arg in enumerate(args):
            value = args[index + 1] if arg == "--output-root" and index + 1 < len(args) else None
            if arg.startswith("--output-root="):
                value = arg.split("=", 1)[1]
            if value and Path(value).resolve() == output_root:
                active.append(pid)
                break
    return active


def scene_complete(result: dict, directory: Path) -> bool:
    return (
        result.get("exit_code") == 0
        and "api_error" not in result
        and not (directory / "run_error.json").exists()
        and (directory / "predictions.json").is_file()
        and (directory / "metrics.json").is_file()
    )


def run_scene(name: str, output_root: Path, model: str, effort: str, max_turns: int, timeout: int, render_threads: int, resume: bool = False, api_backend: str = "openai", api_mode: str = "responses") -> dict:
    directory = output_root / name
    if resume and directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "spatialscout.benchmark.room",
        "--mode", "run", "--catalog-scene", name,
        "--model", model, "--api-mode", api_mode, "--reasoning-effort", effort,
        "--max-turns", str(max_turns), "--timeout", str(timeout),
        "--output-dir", str(directory),
    ]
    if api_backend != "openai":
        command.extend(("--api-backend", api_backend))
    start = time.monotonic()
    env = os.environ.copy()
    env["LP_NUM_THREADS"] = str(render_threads)
    with (directory / "run.log").open("w") as log:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    result = {"scene": name, "model": model, "api_backend": api_backend, "exit_code": process.returncode, "wall_seconds": round(time.monotonic() - start, 1), "output_dir": str(directory)}
    metrics_path = directory / "metrics.json"
    if metrics_path.is_file():
        result["metrics"] = json.loads(metrics_path.read_text())
    usage_path = directory / "token_usage.json"
    if usage_path.is_file():
        result["token_usage"] = {key: value for key, value in json.loads(usage_path.read_text()).items() if key != "turns"}
    error_path = directory / "run_error.json"
    if error_path.is_file():
        error = json.loads(error_path.read_text())
        if error.get("type") == "tool_protocol_error":
            result["tool_error"] = error
        elif error.get("type") == "agent_stopped_without_finish":
            result["agent_error"] = error
        else:
            result["api_error"] = error
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallelism", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--api-backend", choices=("openai", "anthropic", "gemini"), default=os.getenv("VLM_API_BACKEND", "openai"))
    parser.add_argument("--api-mode", choices=("responses", "chat"), default="responses")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--render-threads", type=int, default=int(os.getenv("LP_NUM_THREADS", "1")), help="llvmpipe threads per scene; 1 is faster for concurrent camera agents on this host")
    parser.add_argument("--batch", choices=("original_10", "additional_50", "procthor_30", "prepared_80", "all"), default="original_10")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "room_benchmark" / "astra_high_100")
    parser.add_argument("--resume", action="store_true", help="Continue an existing batch, keeping completed scenes and overwriting failed or unfinished scenes")
    args = parser.parse_args()
    if args.parallelism < 1 or args.max_turns < 1 or args.timeout < 1 or args.render_threads < 1:
        parser.error("parallelism, max-turns, timeout and render-threads must be positive")
    if args.api_backend == "gemini" and args.reasoning_effort not in ("low", "medium", "high"):
        parser.error("Gemini reasoning effort must be low, medium, or high")
    catalog = json.loads(CATALOG_PATH.read_text())["scenes"]
    selected_batches = {"additional_50", "procthor_30"} if args.batch == "prepared_80" else {args.batch}
    scenes = [entry["name"] for entry in catalog if args.batch == "all" or entry.get("batch", "original_10") in selected_batches]
    args.output_root = args.output_root.resolve()
    summary_path = args.output_root / "batch_summary.json"
    if args.resume:
        if not summary_path.is_file():
            parser.error(f"resume requires an existing batch_summary.json: {summary_path}")
    else:
        args.output_root.mkdir(parents=True, exist_ok=True)
    active = active_batch_pids(args.output_root)
    if active:
        parser.error(f"another batch runner is still using this directory (PID {', '.join(map(str, active))}); stop or wait for it before resuming")
    with (args.output_root / ".batch.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"another batch runner is still using {args.output_root}")
        previous = json.loads(summary_path.read_text()) if args.resume else {}
        prior_model = previous.get("original_model", previous.get("model", args.model))
        if args.resume and previous.get("model") != args.model:
            print(f"WARNING: resuming with model {args.model!r}; previous batch model was {previous.get('model')!r}. Results may contain multiple models.", file=sys.stderr, flush=True)
        results = {item["scene"]: dict(item) for item in previous.get("results", [])}
        unknown = set(results) - set(scenes)
        if unknown:
            parser.error(f"resume directory contains scenes outside batch {args.batch}: {sorted(unknown)}")
        for item in results.values():
            item.setdefault("model", prior_model)
            item.setdefault("api_backend", previous.get("api_backend", "openai"))
        completed = {name for name, result in results.items() if scene_complete(result, args.output_root / name)}
        pending = [name for name in scenes if name not in completed]
        model_history = previous.get("models_used", [prior_model] if args.resume else [])
        if args.model not in model_history:
            model_history.append(args.model)
        resume_history = previous.get("resume_history", [])
        if args.resume:
            resume_history.append({"started_utc": datetime.now(timezone.utc).isoformat(), "model": args.model, "api_backend": args.api_backend, "parallelism": args.parallelism, "pending_scenes": len(pending)})

        def save_summary() -> None:
            summary = {
                "batch": args.batch,
                "model": args.model,
                "api_backend": args.api_backend,
                "api_mode": args.api_mode,
                "original_model": prior_model,
                "models_used": model_history,
                "reasoning_effort": args.reasoning_effort,
                "max_turns": args.max_turns,
                "parallelism": args.parallelism,
                "render_threads": args.render_threads,
                "completed_count": len(completed),
                "pending_scenes": [name for name in scenes if name not in completed],
                "resume_history": resume_history,
                "results": [results[name] for name in scenes if name in results],
            }
            temporary = summary_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
            temporary.replace(summary_path)

        save_summary()
        print(f"{len(completed)} completed, {len(pending)} pending in {args.output_root}", flush=True)
        with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
            jobs = {pool.submit(run_scene, name, args.output_root, args.model, args.reasoning_effort, args.max_turns, args.timeout, args.render_threads, args.resume, args.api_backend, args.api_mode): name for name in pending}
            for future in as_completed(jobs):
                result = future.result()
                results[result["scene"]] = result
                if scene_complete(result, args.output_root / result["scene"]):
                    completed.add(result["scene"])
                save_summary()
                metrics = result.get("metrics", {})
                print(
                    f"{result['scene']}: exit={result['exit_code']} predicted={metrics.get('predicted_count')} "
                    f"all_objects_loss_m={metrics.get('full_scene_loss_m')} "
                    f"submitted_objects_loss_m={metrics.get('mean_euclidean_error_m')} "
                    f"tokens={result.get('token_usage', {}).get('total_tokens')} "
                    f"wall={result['wall_seconds']}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
