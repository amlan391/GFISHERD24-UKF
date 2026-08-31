import argparse
import re
from pathlib import Path

import cv2


XC_COL = 1
YC_COL = 2
W_COL = 3
H_COL = 4
CONF_COL = 5
TRACK_ID_COL = 6


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert Ultralytics tracking labels into one MOT-format result file per sequence"
        )
    )

    parser.add_argument(
        "--track-root",
        required=True,
        help="Root directory containing Ultralytics tracking outputs. Each sequence directory must contain labels/",
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root of the prepared MOT-format test dataset. Each sequence must contain img1/",
    )

    parser.add_argument(
        "--output",
        default="runs/mot",
        help="Directory where converted MOT-format result files are written",
    )

    return parser.parse_args()


def natural_frame_key(path: Path) -> int:
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else 10**12


def find_image_size_for_sequence(seq_name: str, dataset_root: Path):
    img_dir = dataset_root / seq_name / "img1"

    if not img_dir.exists():
        return None, None

    image_files = sorted(
        list(img_dir.glob("*.jpg"))
        + list(img_dir.glob("*.jpeg"))
        + list(img_dir.glob("*.png"))
        + list(img_dir.glob("*.bmp"))
        + list(img_dir.glob("*.webp"))
    )

    if not image_files:
        return None, None

    img = cv2.imread(str(image_files[0]))

    if img is None:
        return None, None

    h, w = img.shape[:2]
    return w, h


def yolo_to_xywh(xc, yc, bw, bh, img_w, img_h):
    abs_w = bw * img_w
    abs_h = bh * img_h

    x1 = (xc * img_w) - abs_w / 2.0
    y1 = (yc * img_h) - abs_h / 2.0

    return x1, y1, abs_w, abs_h


def clip_box(x1, y1, w, h, img_w, img_h):
    x2 = x1 + w
    y2 = y1 + h

    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(img_w), x2)
    y2 = min(float(img_h), y2)

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)

    return x1, y1, w, h


def merge_sequence_labels(
    seq_dir: Path,
    dataset_root: Path,
    out_root: Path,
):
    labels_dir = seq_dir / "labels"

    if not labels_dir.exists():
        print(f"Skipping {seq_dir.name}: no labels folder found")
        return

    txt_files = sorted(
        labels_dir.glob("*.txt"),
        key=natural_frame_key,
    )

    if not txt_files:
        print(f"Skipping {seq_dir.name}: no txt files in labels/")
        return

    img_w, img_h = find_image_size_for_sequence(
        seq_dir.name,
        dataset_root,
    )

    if img_w is None or img_h is None:
        print(
            f"Skipping {seq_dir.name}: "
            "could not determine image size"
        )
        return

    merged_lines = []

    for txt_file in txt_files:
        frame_id = natural_frame_key(txt_file)

        with open(txt_file, "r") as f:
            for raw_line in f:
                parts = raw_line.strip().split()

                if len(parts) < 7:
                    continue

                try:
                    xc = float(parts[XC_COL])
                    yc = float(parts[YC_COL])
                    bw = float(parts[W_COL])
                    bh = float(parts[H_COL])
                    conf = float(parts[CONF_COL])
                    track_id = int(float(parts[TRACK_ID_COL]))
                except (ValueError, IndexError):
                    print(
                        f"Warning: parse failed in {txt_file}: "
                        f"{raw_line.strip()}"
                    )
                    continue

                x1, y1, w, h = yolo_to_xywh(
                    xc,
                    yc,
                    bw,
                    bh,
                    img_w,
                    img_h,
                )

                x1, y1, w, h = clip_box(
                    x1,
                    y1,
                    w,
                    h,
                    img_w,
                    img_h,
                )

                # MOT-format output:
                # frame,id,x,y,w,h,score,class,visibility
                merged_lines.append(
                    (
                        f"{frame_id},{track_id},"
                        f"{x1:.2f},{y1:.2f},"
                        f"{w:.2f},{h:.2f},"
                        f"{conf:.6f},1,1\n"
                    )
                )

    out_file = out_root / f"{seq_dir.name}.txt"

    with open(out_file, "w") as f:
        f.writelines(merged_lines)

    print(f"Saved merged file: {out_file}")


def main():
    args = parse_args()

    track_root = Path(args.track_root)
    dataset_root = Path(args.dataset_root)
    out_root = Path(args.output)

    if not track_root.exists():
        raise FileNotFoundError(
            f"Tracking output root not found: {track_root}"
        )

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}"
        )

    out_root.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted(
        path for path in track_root.iterdir() if path.is_dir()
    )

    if not seq_dirs:
        raise RuntimeError(
            f"No sequence directories found in: {track_root}"
        )

    for seq_dir in seq_dirs:
        merge_sequence_labels(
            seq_dir,
            dataset_root,
            out_root,
        )

    print("Done. All MOT files were saved sequence-wise")


if __name__ == "__main__":
    main()
