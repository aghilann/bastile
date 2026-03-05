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

# This is evaluated on your local machine at deploy/build time.
PROJECT_ROOT = Path(__file__).resolve().parent


def _ignore_path(path: Path) -> bool:
    p = Path(path)
    if any(part in EXCLUDED_NAMES for part in p.parts):
        return True
    return p.name.endswith(".egg-info")


# Local files (Modal reads these from your machine when building the image)
pyproject_local = PROJECT_ROOT / "pyproject.toml"
uv_lock_local = PROJECT_ROOT / "uv.lock"
readme_local = PROJECT_ROOT / "README.md"  # optional

image = (
    modal.Image.from_registry(CUDA_BASE, add_python="3.12")
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends bash make git && rm -rf /var/lib/apt/lists/*",
        "python -m pip install -U pip uv",
    )
    # Bake dependency manifests into the image layer for caching
    .add_local_file(str(pyproject_local), f"{REMOTE_PROJECT_DIR}/pyproject.toml", copy=True)
    .add_local_file(str(uv_lock_local), f"{REMOTE_PROJECT_DIR}/uv.lock", copy=True)
)

# If README exists locally, include it so hatchling readme validation doesn't fail
if readme_local.exists():
    image = image.add_local_file(str(readme_local), f"{REMOTE_PROJECT_DIR}/README.md", copy=True)

image = (
    image.run_commands(f"cd {REMOTE_PROJECT_DIR} && uv sync --dev --frozen")
    # Ship the whole repo at container startup (fast iteration, no image rebuild)
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