"""Minimal PyTorch + CUDA smoke test for Iridis X GPU nodes."""

import sys
import torch


def main():
    print("=" * 60)
    print("PyTorch + CUDA Smoke Test")
    print("=" * 60)

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available.")
        sys.exit(1)

    print(f"CUDA version    : {torch.version.cuda}")
    print(f"Device count    : {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {props.name}  |  {props.total_memory / 1024**3:.1f} GB")

    # Quick tensor operation on GPU
    device = torch.device("cuda:0")
    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    c = a @ b
    print(f"\nMatrix multiply (1024x1024) on {torch.cuda.get_device_name(0)}: OK")
    print(f"Result norm: {c.norm().item():.4f}")

    print("\nPASS: All checks succeeded.")


if __name__ == "__main__":
    main()
