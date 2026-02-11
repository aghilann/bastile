.PHONY: bench-small bench-8b bench-8b-liger bench-profile bench-all

# Qwen3-8B comparison: HuggingFace vs Liger vs Bastile (single config)
bench-small:
	python -u -m tests.benchmarks.e2e.comparison_small

# Qwen3-8B seq length sweep: PyTorch vs Liger vs Bastile (parallel on 3 GPUs)
bench-8b:
	python -u -m tests.benchmarks.e2e.qwen_8b_seqlen

# Qwen3-8B seq length sweep: PyTorch vs Liger vs Bastile (sequential, 1 GPU)
bench-8b-seq:
	python -u -m tests.benchmarks.e2e.qwen_8b_seqlen --sequential

# Qwen3-8B seq length sweep: PyTorch vs Liger only
bench-8b-liger:
	python -u -m tests.benchmarks.e2e.qwen_8b_liger

# Kernel-level profiling with torch.profiler
bench-profile:
	python -u -m tests.benchmarks.e2e.profile_kernels

# Run all benchmarks
bench-all: bench-small bench-8b
