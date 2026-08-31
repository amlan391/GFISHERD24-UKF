import argparse
import configparser
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare MOT-format tracking results for MOTMetrics and TrackEval"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing filtered MOT-format result files",
    )

    parser.add_argument(
        "--gt-root",
        required=True,
        help="Root of the prepared MOT-format test dataset containing sequence folders and seqinfo.ini files",
    )

    parser.add_argument(
        "--output",
        default="runs/evaluation",
        help="Directory where evaluation files are created",
    )

    return parser.parse_args()


def read_sequence_lengths(gt_root: Path):
    sequence_lengths = {}

    for ini_path in gt_root.glob("**/seqinfo.ini"):
        config = configparser.ConfigParser()

        try:
            config.read(ini_path)

            seq_name = config.get("Sequence", "name")
            seq_length = config.getint("Sequence", "seqLength")

            sequence_lengths[seq_name] = seq_length

        except (configparser.Error, ValueError, KeyError):
            print(f"Could not read {ini_path}")

    return sequence_lengths


def fix_track_file(track_file: Path, max_frame):
    lines = []
    min_frame = float("inf")

    with open(track_file, "r") as f:
        for line in f:
            stripped = line.strip()

            if not stripped:
                continue

            parts = stripped.split(",")

            if len(parts) < 6:
                print(f"Skipping invalid line in {track_file.name}")
                continue

            try:
                frame_num = int(float(parts[0]))
            except ValueError:
                print(f"Skipping invalid frame number in {track_file.name}")
                continue

            min_frame = min(min_frame, frame_num)
            lines.append((frame_num, stripped))

    if not lines:
        return

    starts_at_zero = min_frame == 0
    has_extra_frames = any(
        frame_num > max_frame
        for frame_num, _ in lines
    )

    fixed_lines = []

    for frame_num, stripped in lines:
        if starts_at_zero:
            frame_num += 1

            parts = stripped.split(",")
            parts[0] = str(frame_num)
            stripped = ",".join(parts)

        if frame_num <= max_frame:
            fixed_lines.append(stripped)

    with open(track_file, "w") as f:
        for line in fixed_lines:
            f.write(line + "\n")

    if starts_at_zero:
        print(f"{track_file.name}: shifted frame numbers from 0-based to 1-based")

    if has_extra_frames:
        print(f"{track_file.name}: removed frames past the sequence length")


def main():
    args = parse_args()

    input_dir = Path(args.input)
    gt_root = Path(args.gt_root)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    if not gt_root.exists():
        raise FileNotFoundError(
            f"Ground truth directory not found: {gt_root}"
        )

    motmetrics_dir = output_dir / "motmetrics"
    trackeval_data_dir = (
        output_dir
        / "trackeval"
        / "botsort"
        / "data"
    )
    seqmap_file = output_dir / "seqmap.txt"

    motmetrics_dir.mkdir(parents=True, exist_ok=True)
    trackeval_data_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.txt"))

    if not input_files:
        raise RuntimeError(
            f"No MOT-format result files found in {input_dir}"
        )

    print(f"Copying {len(input_files)} tracking files to MOTMetrics")

    for input_file in input_files:
        destination = motmetrics_dir / input_file.name
        shutil.copy2(input_file, destination)

    print("Reading sequence lengths")

    sequence_lengths = read_sequence_lengths(gt_root)

    print("Checking frame numbers and sequence lengths")

    for track_file in sorted(motmetrics_dir.glob("*.txt")):
        seq_name = track_file.stem

        max_frame = sequence_lengths.get(seq_name)

        if max_frame is None:
            print(f"No sequence length found for {seq_name}")
            max_frame = float("inf")

        fix_track_file(
            track_file,
            max_frame,
        )

    print("Creating TrackEval links")

    sequence_names = []

    for track_file in sorted(motmetrics_dir.glob("*.txt")):
        seq_name = track_file.stem
        sequence_names.append(seq_name)

        link_path = trackeval_data_dir / track_file.name

        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()

        link_path.symlink_to(track_file.resolve())

    print(f"Creating seqmap at {seqmap_file}")

    with open(seqmap_file, "w") as f:
        f.write("name\n")

        for name in sorted(sequence_names):
            f.write(f"{name}\n")

    print("Evaluation files ready")


if __name__ == "__main__":
    main()
