"""
record_keystrokes.py
=====================
Command-line script for collecting labelled keystroke audio samples.

Usage
-----
    python -m scripts.record_keystrokes --keys a b c space --n 20
    echotype-record --keys a b c --n 30 --output data/raw

The script:
  1. Displays and prompts the user to accept the consent notice.
  2. For each requested key label, plays a prompt and records N audio clips.
  3. Saves WAV files under ``<output>/<label>/<label>_NNNN.wav``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="echotype-record",
        description="Collect labelled keystroke audio clips.",
    )
    parser.add_argument(
        "--keys", nargs="+", default=["a", "b", "c"],
        metavar="KEY",
        help="Key labels to record (default: a b c).",
    )
    parser.add_argument(
        "--n", type=int, default=20, metavar="N",
        help="Number of clips to record per key (default: 20).",
    )
    parser.add_argument(
        "--output", default="data/raw", metavar="DIR",
        help="Root output directory (default: data/raw).",
    )
    parser.add_argument(
        "--pause", type=float, default=1.5, metavar="SECS",
        help="Pause in seconds between recordings (default: 1.5).",
    )
    parser.add_argument(
        "--delete-data", action="store_true",
        help="Delete the output directory and exit.",
    )
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="Skip consent prompt (for automated testing only).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    output_dir = Path(args.output)

    # ------------------------------------------------------------------
    # Handle --delete-data
    # ------------------------------------------------------------------
    if args.delete_data:
        if output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"Deleted data directory: {output_dir}")
        else:
            print(f"Data directory does not exist: {output_dir}")
        return 0

    # ------------------------------------------------------------------
    # Consent
    # ------------------------------------------------------------------
    from echotype_lab.consent import request_consent  # noqa: PLC0415

    accepted = request_consent(non_interactive=args.non_interactive)
    if not accepted:
        return 1

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    try:
        from echotype_lab.audio.recorder import KeystrokeRecorder  # noqa: PLC0415
    except ImportError as exc:
        logger.error("Could not import recorder: %s", exc)
        logger.error("Is pyaudio installed?  pip install pyaudio")
        return 1

    total = len(args.keys) * args.n
    print(f"\nWill record {args.n} clips × {len(args.keys)} keys = {total} total clips.")
    print(f"Output directory: {output_dir.resolve()}\n")

    for key in args.keys:
        print(f"\n─── Recording key: '{key}' ───")
        rec = KeystrokeRecorder(output_dir=output_dir, label=key)
        paths = rec.record_n_keystrokes(args.n, pause=args.pause)
        print(f"  Saved {len(paths)} clips for '{key}'.")

    print("\nRecording complete.  Run echotype-train to train a classifier.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
