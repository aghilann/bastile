"""Minimal cuTILE kernel test to debug compilation."""
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

@ct.kernel
def copy_kernel(x, out, TILE_SIZE: ConstInt):
    """Simplest possible kernel: copy input to output."""
    row = ct.bid(0)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)
    xj = ct.gather(x, (row, offsets), check_bounds=True, padding_value=0.0)
    ct.scatter(out, (row, offsets), xj, check_bounds=True)


if __name__ == "__main__":
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Capability: {torch.cuda.get_device_capability()}")
    print("Launching copy kernel...")

    ct.launch(
        torch.cuda.current_stream(),
        (4,),
        copy_kernel,
        (x, out, 128),
    )
    torch.cuda.synchronize()

    print(f"Max diff: {(x - out).abs().max().item()}")
    print("SUCCESS!")
