export LD_LIBRARY_PATH := /usr/local/cuda-13.0/compat:/usr/lib/x86_64-linux-gnu:$(LD_LIBRARY_PATH)

.PHONY: bench-small bench-8b bench-8b-liger bench-profile bench-all

# Qwen3-8B comparison: HuggingFace vs Liger vs Bastile (single config)
bench-small:
	uv run python3 -u -m tests.benchmarks.e2e.comparison_small

# Qwen3-8B seq length sweep: PyTorch vs Liger vs Bastile (parallel on 3 GPUs)
bench-8b:
	uv run python3 -u -m tests.benchmarks.e2e.qwen_8b_seqlen

# Qwen3-8B seq length sweep: PyTorch vs Liger vs Bastile (sequential, 1 GPU)
bench-8b-seq:
	uv run python3 -u -m tests.benchmarks.e2e.qwen_8b_seqlen --sequential

# Qwen3-8B seq length sweep: PyTorch vs Liger only
bench-8b-liger:
	uv run python3 -u -m tests.benchmarks.e2e.qwen_8b_liger

# Kernel-level profiling with torch.profiler
bench-profile:
	uv run python3 -u -m tests.benchmarks.e2e.profile_kernels

# Kernel micro-benchmark: Quack vs CuTile RMSNorm
bench-rmsnorm:
	uv run python3 -u -m tests.benchmarks.kernel.rms_norm_quack_vs_cutile

# GPT-OSS-20B seq length sweep: PyTorch vs Liger vs Bastile (parallel on 3 GPUs)
bench-gpt-oss:
	uv run python3 -u -m tests.benchmarks.e2e.gpt_oss_20b_seqlen

# GPT-OSS-20B seq length sweep: PyTorch vs Liger vs Bastile (sequential, 1 GPU)
bench-gpt-oss-seq:
	uv run python3 -u -m tests.benchmarks.e2e.gpt_oss_20b_seqlen --sequential

# Run all benchmarks
bench-all: bench-small bench-8b bench-gpt-oss
