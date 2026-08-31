import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BoT-SORT tracking on all GFISHERD24 test sequences"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained YOLO detector checkpoint (best.pt)",
    )

    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to the prepared test sequence root. Each sequence must contain an img1/ directory",
    )

    parser.add_argument(
        "--tracker",
        default="configs/custom_botsort.yaml",
        help="Path to the custom BoT-SORT tracker configuration",
    )

    parser.add_argument(
        "--output",
        default="runs/tracking",
        help="Directory where Ultralytics tracking outputs are saved",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Detection confidence threshold",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)
    root = Path(args.data_root)

    if not root.exists():
        raise FileNotFoundError(f"Test sequence root not found: {root}")

    sequence_dirs = sorted(
        path for path in root.iterdir() if path.is_dir()
    )

    if not sequence_dirs:
        raise RuntimeError(f"No sequence directories found in: {root}")

    for seq_dir in sequence_dirs:
        img_dir = seq_dir / "img1"

        if not img_dir.exists():
            print(f"Skipping {seq_dir.name}: no img1 folder found")
            continue

        print(f"Tracking sequence: {seq_dir.name}")

        model.track(
            source=str(img_dir),
            tracker=args.tracker,
            persist=True,
            save=False,
            save_txt=True,
            save_conf=True,
            conf=args.conf,
            agnostic_nms=True,
            project=args.output,
            name=seq_dir.name,
            exist_ok=True,
            workers=0,
        )

    print("Tracking completed for all sequences")


if __name__ == "__main__":
    main()
