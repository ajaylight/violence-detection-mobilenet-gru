from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

DEFAULT_HANDLE = "mohamedmustafa/real-life-violence-situations-dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Kaggle violence dataset.")
    parser.add_argument(
        "--output-dir",
        default="dataset/kaggle",
        help="Directory where KaggleHub or Kaggle CLI should place the downloaded dataset.",
    )
    parser.add_argument(
        "--handle",
        default=DEFAULT_HANDLE,
        help="Kaggle dataset handle.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub

        path = kagglehub.dataset_download(args.handle, output_dir=str(output_dir))
        print(f"Path to dataset files: {path}")
        return
    except Exception:
        kaggle_cli = shutil.which("kaggle")
        if kaggle_cli:
            cmd = [kaggle_cli, "datasets", "download", "-d", args.handle, "-p", str(output_dir), "--unzip"]
            print("`kagglehub` unavailable — using Kaggle CLI:", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
                print("Downloaded dataset to:", output_dir)
                return
            except subprocess.CalledProcessError as err:
                print("Kaggle CLI failed:", err, file=sys.stderr)
                raise
        else:
            print(
                "kagglehub is not installed and the Kaggle CLI was not found.\n"
                "Install one of them, for example: `pip install kagglehub`\n"
                "or follow Kaggle CLI setup: https://github.com/Kaggle/kaggle-api",
                file=sys.stderr,
            )
            raise


if __name__ == "__main__":
    main()

