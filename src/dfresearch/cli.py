"""CLI entry point for dfresearch."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="dfresearch",
        description="Autonomous deepfake detection research for BitMind Subnet 34",
    )
    subparsers = parser.add_subparsers(dest="command")

    # prepare
    prep = subparsers.add_parser("prepare", help="Download and prepare datasets")
    prep.add_argument("--modality", choices=["image", "video", "audio", "all"], default="all")
    prep.add_argument("--verify", action="store_true")

    # train
    train = subparsers.add_parser("train", help="Run training")
    train.add_argument("--modality", required=True, choices=["image", "video", "audio"])
    train.add_argument("--model", default=None)

    # evaluate
    ev = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    ev.add_argument("--modality", required=True, choices=["image", "video", "audio"])
    ev.add_argument("--model", default=None)
    ev.add_argument("--weights", default=None)

    # export
    exp = subparsers.add_parser("export", help="Export model for competition")
    exp.add_argument("--modality", required=True, choices=["image", "video", "audio"])
    exp.add_argument("--model", required=True)
    exp.add_argument("--weights", default=None)

    args = parser.parse_args()

    if args.command == "prepare":
        from prepare import download_datasets, verify_cache
        modalities = ["image", "video", "audio"] if args.modality == "all" else [args.modality]
        for mod in modalities:
            if args.verify:
                verify_cache(mod)
            else:
                download_datasets(mod)

    elif args.command == "train":
        import subprocess
        script = f"train_{args.modality}.py"
        cmd = [sys.executable, script]
        if args.model:
            cmd.extend(["--model", args.model])
        subprocess.run(cmd)

    elif args.command == "evaluate":
        import subprocess
        cmd = [sys.executable, "evaluate.py", "--modality", args.modality]
        if args.model:
            cmd.extend(["--model", args.model])
        if args.weights:
            cmd.extend(["--weights", args.weights])
        subprocess.run(cmd)

    elif args.command == "export":
        import subprocess
        cmd = [sys.executable, "export.py", "--modality", args.modality, "--model", args.model]
        if args.weights:
            cmd.extend(["--weights", args.weights])
        subprocess.run(cmd)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
