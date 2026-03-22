#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Unified lifecycle inventory review entrypoint")
    parser.add_argument("--profile", choices=["process", "lifecyclemodel"], default="process")
    parser.add_argument("--run-root")
    parser.add_argument("--run-id")
    parser.add_argument("--out-dir")
    parser.add_argument("--start-ts")
    parser.add_argument("--end-ts")
    parser.add_argument("--logic-version")
    parser.add_argument("--enable-llm", action="store_true")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-max-processes", type=int)
    args = parser.parse_args()

    if args.profile == "process":
        required = ["run_root", "run_id", "out_dir"]
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            parser.error("process profile requires: --run-root --run-id --out-dir")

        target = Path(__file__).resolve().parents[1] / "profiles" / "process" / "scripts" / "run_process_review.py"
        cmd = [
            sys.executable,
            str(target),
            "--run-root", args.run_root,
            "--run-id", args.run_id,
            "--out-dir", args.out_dir,
        ]
        if args.start_ts:
            cmd += ["--start-ts", args.start_ts]
        if args.end_ts:
            cmd += ["--end-ts", args.end_ts]
        if args.logic_version:
            cmd += ["--logic-version", args.logic_version]
        if args.enable_llm is True:
            cmd += ["--enable-llm"]
        if args.llm_model:
            cmd += ["--llm-model", args.llm_model]
        if args.llm_max_processes is not None:
            cmd += ["--llm-max-processes", str(args.llm_max_processes)]
        raise SystemExit(subprocess.call(cmd))

    print(
        f"Profile '{args.profile}' not implemented yet. "
        "Next step: add reviewer logic under profiles/"
        f"{args.profile}/ and wire it in scripts/run_review.py."
    )
    return 0


if __name__ == "__main__":
    main()
