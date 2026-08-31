import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter MOT-format detections/tracks by bounding box area"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing MOT-format .txt files",
    )

    parser.add_argument(
        "--output",
        default="runs/mot_filtered",
        help="Directory where filtered MOT files are written",
    )

    parser.add_argument(
        "--min-area",
        type=float,
        default=800.0,
        help="Minimum bounding box area to keep",
    )

    return parser.parse_args()


def filter_file(input_file: Path, output_file: Path, min_area: float):
    kept = 0
    removed = 0

    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) < 6:
                print(f"Warning: skipping malformed line in {input_file.name}")
                continue

            try:
                width = float(parts[4])
                height = float(parts[5])
            except ValueError:
                print(f"Warning: skipping malformed line in {input_file.name}")
                continue

            area = width * height

            if area >= min_area:
                f_out.write(line + "\n")
                kept += 1
            else:
                removed += 1

    print(
        f"{input_file.name}: "
        f"kept={kept}, removed={removed}"
    )


def main():
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.txt"))

    if not input_files:
        raise RuntimeError(
            f"No MOT-format .txt files found in: {input_dir}"
        )

    for input_file in input_files:
        output_file = output_dir / input_file.name

        filter_file(
            input_file=input_file,
            output_file=output_file,
            min_area=args.min_area,
        )


if __name__ == "__main__":
    main()
