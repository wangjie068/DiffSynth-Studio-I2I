import argparse
import os
import subprocess
import time


def query_gpu_utilization(gpu: str) -> int | None:
    command = [
        "nvidia-smi",
        f"--id={gpu}",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return int(lines[0])
    except ValueError:
        return None


def burn_gpu(gpu: str, seconds: float, matrix_size: int, dtype_name: str) -> None:
    # Import torch lazily so the script can still print useful errors on machines
    # where nvidia-smi exists but the Python env is not the inference env.
    import torch

    device = torch.device(f"cuda:{gpu}")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]

    with torch.no_grad():
        a = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
        b = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
        end_time = time.time() + seconds
        while time.time() < end_time:
            c = a @ b
            # Touch the result so kernels are actually scheduled.
            _ = c[0, 0].item()
        del a, b, c
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep allocated GPUs non-idle during model-loading phases by running "
            "short low-memory compute bursts when utilization is below a threshold."
        )
    )
    parser.add_argument("--gpus", nargs="+", default=["0"], help="GPU ids visible to this process.")
    parser.add_argument("--threshold", type=int, default=3, help="Burn only when utilization <= this value.")
    parser.add_argument("--check-interval", type=float, default=5.0, help="Seconds between checks.")
    parser.add_argument("--burn-seconds", type=float, default=2.0, help="Duration of each keepalive burst.")
    parser.add_argument("--matrix-size", type=int, default=2048, help="Matrix size for the matmul workload.")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Use float16 by default to keep memory low.",
    )
    parser.add_argument("--once", action="store_true", help="Run one check cycle and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "GPU keepalive started: "
        f"gpus={args.gpus}, threshold={args.threshold}%, "
        f"check_interval={args.check_interval}s, burn_seconds={args.burn_seconds}s, "
        f"matrix_size={args.matrix_size}, dtype={args.dtype}",
        flush=True,
    )
    print("Use Ctrl+C to stop.", flush=True)
    while True:
        for gpu in args.gpus:
            utilization = query_gpu_utilization(gpu)
            if utilization is None:
                print(f"[gpu {gpu}] could not query utilization; skipping", flush=True)
                continue
            if utilization <= args.threshold:
                print(f"[gpu {gpu}] util={utilization}% <= {args.threshold}%, burn", flush=True)
                try:
                    burn_gpu(gpu, args.burn_seconds, args.matrix_size, args.dtype)
                except RuntimeError as error:
                    print(f"[gpu {gpu}] burn failed: {error}", flush=True)
            else:
                print(f"[gpu {gpu}] util={utilization}%, idle keeper sleeps", flush=True)
        if args.once:
            break
        time.sleep(args.check_interval)


if __name__ == "__main__":
    main()
