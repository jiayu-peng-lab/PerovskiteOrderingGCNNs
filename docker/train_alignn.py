import argparse
import os
import json

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Only print versions and exit")
    parser.add_argument("--data", type=str, default="/workspace/data", help="Dataset root inside container")
    parser.add_argument("--config", type=str, default=None, help="ALIGNN config YAML (optional)")
    parser.add_argument("--out", type=str, default="/workspace/output", help="Output directory")
    args = parser.parse_args()

    if args.check:
        import dgl
        import alignn
        print(json.dumps({
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "num_gpus": torch.cuda.device_count(),
            "dgl": dgl.__version__,
            "alignn": alignn.__version__ if hasattr(alignn, "__version__") else "unknown",
        }, indent=2))
        return

    os.makedirs(args.out, exist_ok=True)

    # Minimal example: call ALIGNN training entrypoint programmatically if available.
    # Otherwise, instruct user to run alignn-train via CLI.
    try:
        from alignn.train import train
    except Exception:
        print("alignn.train not available; use the CLI inside the container, e.g.:\n"
              "  alignn-train --root {} --config {} --output {}".format(args.data, args.config, args.out))
        return

    train(root=args.data, config=args.config, output_dir=args.out)


if __name__ == "__main__":
    main()


