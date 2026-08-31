import argparse
import sys
from pathlib import Path

import motmetrics as mm
import numpy as np


# Compatibility fix for newer NumPy versions
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MOTMetrics and TrackEval on tracking results"
    )

    parser.add_argument(
        "--gt-root",
        required=True,
        help="Root of the prepared MOT-format test dataset containing sequence folders and ground truth",
    )

    parser.add_argument(
        "--evaluation-root",
        required=True,
        help="Directory created by prepare_evaluation.py",
    )

    parser.add_argument(
        "--trackeval-root",
        required=True,
        help="Path to the TrackEval repository",
    )

    parser.add_argument(
        "--iou-distance",
        type=float,
        default=0.8,
        help="Maximum IoU distance used by MOTMetrics",
    )

    return parser.parse_args()


def run_motmetrics(gt_root: Path, evaluation_root: Path, iou_distance: float):
    print("\n=== MOTMETRICS RESULTS ===")

    mm.lap.default_solver = "lap"

    results_root = evaluation_root / "motmetrics"

    gt_files = {
        path.parents[1].name: path
        for path in gt_root.glob("*/*/gt.txt")
    }

    result_files = {
        path.stem: path
        for path in results_root.glob("*.txt")
    }

    common = sorted(set(gt_files) & set(result_files))
    missing_gt = sorted(set(result_files) - set(gt_files))
    missing_results = sorted(set(gt_files) - set(result_files))

    if missing_gt:
        print(
            "No ground truth found for: "
            + ", ".join(missing_gt)
        )

    if missing_results:
        print(
            "No tracking result found for: "
            + ", ".join(missing_results)
        )

    if not common:
        raise RuntimeError("No matching ground truth and result files found")

    print(f"Evaluating {len(common)} sequences with MOTMetrics")

    gts = {
        name: mm.io.loadtxt(
            str(gt_files[name]),
            fmt="mot15-2D",
            min_confidence=1,
        )
        for name in common
    }

    tracks = {
        name: mm.io.loadtxt(
            str(result_files[name]),
            fmt="mot15-2D",
            min_confidence=-1,
        )
        for name in common
    }

    accumulators = []
    names = []

    for name in common:
        print(f"Comparing {name}")

        accumulator = mm.utils.compare_to_groundtruth(
            gts[name],
            tracks[name],
            "iou",
            distth=iou_distance,
        )

        accumulators.append(accumulator)
        names.append(name)

    metric_handler = mm.metrics.create()

    metrics = [
        "recall",
        "precision",
        "num_unique_objects",
        "mostly_tracked",
        "partially_tracked",
        "mostly_lost",
        "num_false_positives",
        "num_misses",
        "num_switches",
        "num_fragmentations",
        "num_matches",
        "mota",
        "motp",
        "num_objects",
    ]

    summary = metric_handler.compute_many(
        accumulators,
        names=names,
        metrics=metrics,
        generate_overall=True,
    )

    normalization = {
        "num_objects": [
            "num_false_positives",
            "num_misses",
            "num_switches",
            "num_fragmentations",
        ],
        "num_unique_objects": [
            "mostly_tracked",
            "partially_tracked",
            "mostly_lost",
        ],
    }

    for denominator, metric_names in normalization.items():
        for metric_name in metric_names:
            summary[metric_name] = (
                summary[metric_name] / summary[denominator]
            )

    formatters = metric_handler.formatters.copy()

    for metric_name in (
        "num_false_positives",
        "num_misses",
        "num_switches",
        "num_fragmentations",
        "mostly_tracked",
        "partially_tracked",
        "mostly_lost",
    ):
        formatters[metric_name] = formatters["mota"]

    print(
        mm.io.render_summary(
            summary,
            formatters=formatters,
            namemap=mm.io.motchallenge_metric_names,
        )
    )


def run_trackeval(
    gt_root: Path,
    evaluation_root: Path,
    trackeval_root: Path,
):
    print("\n=== TRACKEVAL RESULTS ===")

    sys.path.insert(0, str(trackeval_root))

    try:
        from trackeval import Evaluator, datasets, metrics as trackeval_metrics
    except ImportError as exc:
        raise ImportError(
            f"Could not import TrackEval from {trackeval_root}"
        ) from exc

    seqmap_file = evaluation_root / "seqmap.txt"
    trackers_folder = evaluation_root / "trackeval"

    if not seqmap_file.exists():
        raise FileNotFoundError(
            f"seqmap file not found: {seqmap_file}"
        )

    if not trackers_folder.exists():
        raise FileNotFoundError(
            f"TrackEval directory not found: {trackers_folder}"
        )

    dataset_config = {
        "GT_FOLDER": str(gt_root),
        "TRACKERS_FOLDER": str(trackers_folder),
        "BENCHMARK": "GFISHERD24",
        "SPLIT_TO_EVAL": "",
        "SEQMAP_FILE": str(seqmap_file),
        "DO_PREPROC": False,
        "SKIP_SPLIT_FOL": True,
        "TRACKERS_TO_EVAL": ["botsort"],
        "TRACKER_SUB_FOLDER": "data",
    }

    print(f"Using seqmap: {seqmap_file}")
    print("Evaluating BoT-SORT with TrackEval")

    evaluator = Evaluator()

    dataset_list = [
        datasets.MotChallenge2DBox(dataset_config)
    ]

    metrics_list = [
        trackeval_metrics.HOTA(),
        trackeval_metrics.CLEAR(),
        trackeval_metrics.Identity(),
    ]

    evaluator.evaluate(
        dataset_list,
        metrics_list,
    )


def main():
    args = parse_args()

    gt_root = Path(args.gt_root)
    evaluation_root = Path(args.evaluation_root)
    trackeval_root = Path(args.trackeval_root)

    if not gt_root.exists():
        raise FileNotFoundError(
            f"Ground truth directory not found: {gt_root}"
        )

    if not evaluation_root.exists():
        raise FileNotFoundError(
            f"Evaluation directory not found: {evaluation_root}"
        )

    if not trackeval_root.exists():
        raise FileNotFoundError(
            f"TrackEval directory not found: {trackeval_root}"
        )

    run_motmetrics(
        gt_root=gt_root,
        evaluation_root=evaluation_root,
        iou_distance=args.iou_distance,
    )

    run_trackeval(
        gt_root=gt_root,
        evaluation_root=evaluation_root,
        trackeval_root=trackeval_root,
    )


if __name__ == "__main__":
    main()
