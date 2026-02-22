# modal_run.py
import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "bastile-local-runner"
CUDA_BASE = "nvidia/cuda:13.1.0-devel-ubuntu22.04"

REMOTE_PROJECT_DIR = "/workspace/project"
GPU = "B200"
TIMEOUT_S = 60 * 60
DEFAULT_CMD = "make bench-lce"

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
PROJECT_ROOT = Path(__file__).resolve().parent


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
        f"cd {REMOTE_PROJECT_DIR} && uv sync --dev --frozen",
    )
    # Ship full source at container start (fast iteration, no image rebuild)
    .add_local_dir(
        local_path=str(PROJECT_ROOT),
        remote_path=REMOTE_PROJECT_DIR,
        ignore=_ignore_path,
        copy=False,
    )
)


@app.function(image=image, gpu=GPU, timeout=TIMEOUT_S)
def run(cmd: str) -> None:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")

    subprocess.run(
        ["bash", "-lc", f"cd {REMOTE_PROJECT_DIR} && uv run {cmd}"],
        check=True,
        env=env,
    )


@app.local_entrypoint()
def main(cmd: str = DEFAULT_CMD) -> None:
    with modal.enable_output():
        run.remote(cmd=cmd)