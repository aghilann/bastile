# export LD_LIBRARY_PATH := /usr/local/cuda-13.0/compat:/usr/lib/x86_64-linux-gnu:$(LD_LIBRARY_PATH)
export CUDA_VISIBLE_DEVICES ?= 2

.PHONY: bench-8b bench-8b-seq bench-fsdp bench-rmsnorm bench-lce bench-all test lint fmt

# Qwen3-8B seq length sweep: PyTorch vs Liger vs Bastile (parallel on 3 GPUs)
bench-8b:
	uv run python3 -u -m tests.benchmarks.e2e.qwen_8b_seqlen

# Qwen3-8B seq length sweep: PyTorch vs Liger vs Bastile (sequential, 1 GPU)
bench-8b-seq:
	uv run python3 -u -m tests.benchmarks.e2e.qwen_8b_seqlen --sequential

# Qwen3-8B FSDP multi-GPU benchmark
bench-fsdp:
	uv run python3 -u -m tests.benchmarks.e2e.qwen_8b_fsdp

# Kernel micro-benchmark: RMSNorm
bench-rmsnorm:
	uv run python3 -u -m tests.benchmarks.kernel.rms_norm

# Kernel micro-benchmark: Fused Linear Cross-Entropy
bench-lce:
	uv run python3 -u -m tests.benchmarks.kernel.bench_fused_lce

# Run all benchmarks
bench-all: bench-8b bench-fsdp

# Run ops unit tests
test:
	uv run python3 -m tests.ops.run_all

# Lint and format
lint:
	uv run ruff check src tests

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests
