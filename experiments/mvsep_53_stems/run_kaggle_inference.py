#!/usr/bin/env python3
"""Run the upstream MSST MVSEP 53-stem model from a Kaggle/Python environment.

This is a thin runner around the upstream inference.py. It does not download
weights and does not claim to split lead guitar from rhythm guitar.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CONFIG_RELATIVE = Path(
    "pretrained/mvsep_53_stems/"
    "mvsep_mega_model_bs_roformer_53_stems.yaml"
)
CHECKPOINT_RELATIVE = Path(
    "pretrained/mvsep_53_stems/"
    "mvsep_mega_model_bs_roformer_53_stems_v1.ckpt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MVSEP 53-stem BS-RoFormer using the upstream MSST repository."
    )
    parser.add_argument(
        "--msst-dir",
        type=Path,
        default=Path("/kaggle/working/Music-Source-Separation-Training"),
        help="Path to the cloned Music-Source-Separation-Training repository.",
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        default=None,
        help="Folder containing input audio. Defaults to <msst-dir>/input_test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for separated stems. Defaults to <msst-dir>/separation_results.",
    )
    parser.add_argument(
        "--device-ids",
        default="0",
        help='Device IDs passed to MSST, e.g. "0" or "0,1". Default: "0".',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    msst_dir = args.msst_dir.expanduser().resolve()
    input_dir = (args.input_folder or (msst_dir / "input_test")).expanduser().resolve()
    output_dir = (args.output_dir or (msst_dir / "separation_results")).expanduser().resolve()

    inference_script = msst_dir / "inference.py"
    config_path = msst_dir / CONFIG_RELATIVE
    checkpoint_path = msst_dir / CHECKPOINT_RELATIVE

    required_paths = {
        "MSST inference.py": inference_script,
        "53-stem config": config_path,
        "53-stem checkpoint": checkpoint_path,
        "input folder": input_dir,
    }
    missing = [f"{label}: {path}" for label, path in required_paths.items() if not path.exists()]
    if missing:
        print("Missing required files or folders:", file=sys.stderr)
        for entry in missing:
            print(f"  - {entry}", file=sys.stderr)
        print(
            "\nCheck the upstream repository, config/weight download, and input path.",
            file=sys.stderr,
        )
        return 2

    audio_files = [
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    ]
    if not audio_files:
        print(f"No supported audio files found in: {input_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    # MSST has several configurations/architectures; use the known-working
    # bs_roformer command and config for the tested MVSEP 53-stem checkpoint.
    command = [
        sys.executable,
        str(inference_script),
        "--model_type",
        "bs_roformer",
        "--config_path",
        str(config_path),
        "--start_check_point",
        str(checkpoint_path),
        "--input_folder",
        str(input_dir),
        "--store_dir",
        str(output_dir),
        "--device_ids",
        args.device_ids,
        "--disable_detailed_pbar",
    ]

    print("MSST repository :", msst_dir)
    print("Input folder    :", input_dir)
    print("Output folder   :", output_dir)
    print("Device IDs      :", args.device_ids)
    print("Command         :", " ".join(command), flush=True)

    try:
        subprocess.run(command, cwd=msst_dir, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"MSST inference failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode or 1

    print(f"\nInference finished. Check output files under: {output_dir}")
    print(
        "Note: this checkpoint outputs 53 class stems. "
        "It does not natively produce independent lead/rhythm guitar stems."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
