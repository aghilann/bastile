# export LD_LIBRARY_PATH := /usr/local/cuda-13.0/compat:/usr/lib/x86_64-linux-gnu:$(LD_LIBRARY_PATH)

.PHONY: check-cuda bench-8b bench-0.6b bench-3.5-9b bench-3.5-8b-config bench-rmsnorm bench-lce bench-all test lint fmt

check-cuda:
	uv run python3 -c "import torch; assert torch.backends.cuda.is_built(), 'PyTorch was installed without CUDA support in this environment.'; assert torch.cuda.is_available(), 'CUDA is not available to PyTorch. If you are running locally, verify the NVIDIA driver is loaded and avoid setting CUDA_VISIBLE_DEVICES to a non-existent GPU.'; print(f'Using CUDA device 0: {torch.cuda.get_device_name(0)}')"

# Qwen3-8B seq length sweep: PyTorch vs Liger vs Bastile (single GPU)
bench-3-8b:
	uv run python3 -u -m tests.benchmarks.e2e.qwen3_seqlen --model-size 8b

# Qwen3-0.6B seq length sweep: PyTorch vs Liger vs Bastile (sequential, 1 GPU)
bench-0.6b:
	uv run python3 -u -m tests.benchmarks.e2e.qwen3_seqlen --model-size 0.6b

# Qwen3.5-9B seq length sweep: PyTorch vs Liger vs Bastile (sequential, 1 GPU)
bench-3.5-9b:
	uv run python3 -u -m tests.benchmarks.e2e.qwen3_5_seqlen --model-size 9b

# Qwen3.5-8B seq length sweep via HF config JSON
bench-3.5-8b-config:
	@test -n "$(CONFIG_JSON)" || (echo "Set CONFIG_JSON=/path/to/config.json"; exit 1)
	uv run python3 -u -m tests.benchmarks.e2e.qwen3_5_seqlen --model-size 8b --config-json "$(CONFIG_JSON)"

# Kernel micro-benchmark: RMSNorm
bench-rmsnorm: check-cuda
	uv run python3 -u -m tests.benchmarks.kernel.rms_norm

# Kernel micro-benchmark: Fused Linear Cross-Entropy
bench-lce:
	uv run python3 -u -m tests.benchmarks.kernel.bench_fused_lce

# Run all benchmarks
bench-all: bench-8b bench-0.6b bench-3.5-9b

# Run ops unit tests
test:
	uv run python3 -m tests.ops.run_all

# Lint and format
lint:
	uv run ruff check src tests

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests
