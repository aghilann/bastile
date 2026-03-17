# modal_run.py
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
import re

import modal

APP_NAME = "bastile-local-runner"
CUDA_BASE = "nvidia/cuda:13.1.0-devel-ubuntu22.04"
BENCHMARKS_VOLUME_NAME = "bastile-benchmarks"

REMOTE_PROJECT_DIR = "/workspace/project"
REMOTE_BENCHMARKS_DIR = "/vol/benchmarks"
GPU = "B200"
TIMEOUT_S = 60 * 60
DEFAULT_CMD = "make bench-3.5-9b"

EXCLUDED_NAMES = {
    ".venv",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
    ".DS_Store",
    "node_modules",
}

app = modal.App(APP_NAME)
benchmarks_volume = modal.Volume.from_name(BENCHMARKS_VOLUME_NAME, create_if_missing=True)


def _find_project_root() -> Path:
    candidates = (
        Path(__file__).resolve().parent,
        Path(REMOTE_PROJECT_DIR),
        Path.cwd(),
    )
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(
        "Expected pyproject.toml in one of: "
        + ", ".join(str(candidate / "pyproject.toml") for candidate in candidates)
    )


PROJECT_ROOT = _find_project_root()


def _ignore_path(path: Path) -> bool:
    p = Path(path)
    if any(part in EXCLUDED_NAMES for part in p.parts):
        return True
    return p.name.endswith(".egg-info")


def _must_exist(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"Expected file not found: {p}")
    return p


PYPROJECT = _must_exist(PROJECT_ROOT / "pyproject.toml")
UV_LOCK = _must_exist(PROJECT_ROOT / "uv.lock")
README = PROJECT_ROOT / "README.md"  # optional, but needed if pyproject declares it

image = (
    modal.Image.from_registry(CUDA_BASE, add_python="3.12")
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends bash make git && rm -rf /var/lib/apt/lists/*",
        "python -m pip install -U pip uv",
    )
    .add_local_file(str(PYPROJECT), f"{REMOTE_PROJECT_DIR}/pyproject.toml", copy=True)
    .add_local_file(str(UV_LOCK), f"{REMOTE_PROJECT_DIR}/uv.lock", copy=True)
    # If your pyproject.toml declares readme="README.md", hatchling requires it at build time.
    .add_local_file(str(README), f"{REMOTE_PROJECT_DIR}/README.md", copy=True)
    .run_commands(
        f"cd {REMOTE_PROJECT_DIR} && uv sync --dev --extra bench --frozen",
    )
    # Ship full source at container start (fast iteration, no image rebuild)
    .add_local_dir(
        local_path=str(PROJECT_ROOT),
        remote_path=REMOTE_PROJECT_DIR,
        ignore=_ignore_path,
        copy=False,
    )
)


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_S,
    volumes={REMOTE_BENCHMARKS_DIR: benchmarks_volume},
)
def run(cmd: str) -> None:
    run_stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    cmd_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", cmd).strip("-")[:80] or "command"
    benchmark_dir = f"{REMOTE_BENCHMARKS_DIR}/{run_stamp}-{cmd_slug}"

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
    env["BASTILE_BENCHMARK_DIR"] = benchmark_dir

    try:
        subprocess.run(["bash", "-lc", f"mkdir -p {benchmark_dir}"], check=True, env=env)
        print(f"Writing benchmark outputs to {benchmark_dir}")
        subprocess.run(
            ["bash", "-lc", f"cd {REMOTE_PROJECT_DIR} && {cmd}"],
            check=True,
            env=env,
        )
    finally:
        benchmarks_volume.commit()


@app.local_entrypoint()
def main(cmd: str = DEFAULT_CMD) -> None:
    with modal.enable_output():
        run.remote(cmd=cmd)
