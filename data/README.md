# GFISHERD24 Dataset

The experiments in this repository use the GFISHERD24 underwater fish dataset.

## Public Dataset Resources

The GFISHERD24 source videos are publicly available through Google Cloud:

[GFISHERD24 Source Videos](https://console.cloud.google.com/storage/browser/nmfs_odp_sefsc/PEMD/Gulf%20of%20Mexico%20Reef%20Fish%20Annotated%20Library/For_Training?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=true)

The annotation files used in preparing the dataset are also publicly available:

- [Training annotations](https://storage.googleapis.com/nmfs_odp_hq/nodd_tools/datasets/gfisher/train_annotations_worms.json)
- [Validation annotations](https://storage.googleapis.com/nmfs_odp_hq/nodd_tools/datasets/gfisher/val_annotations_worms_raritystratified.json)
- [Test annotations](https://storage.googleapis.com/nmfs_odp_hq/nodd_tools/datasets/gfisher/train_annotations_worms_raritystratified.json)

## Data Used in the Experiments

Detector training used a separately prepared collection of extracted image frames and corresponding annotations from GFISHERD24.

The experiments used a 75:15:10 train/validation/test split and evaluated two detector settings:

- Single-class, with all fish assigned to one `Fish` class
- Multi-class, using 155 fish classes

The exact image collection and preprocessing steps used in the experiments are not included in this repository. The source videos and annotation files above therefore do not by themselves reproduce the exact detector training dataset used in the paper.

## YOLO Training Dataset

The YOLOv26-L detectors were trained using extracted image frames and YOLO-format bounding box annotations.

The prepared YOLO training dataset follows a structure such as:

```text
GFISHERD24/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

The repository provides two Ultralytics dataset configuration files:

```text
configs/GFISHERD24_single.yaml
configs/GFISHERD24_multi.yaml
```

The single-class configuration uses one class:

```text
0: Fish
```

The multi-class configuration uses the 155 fish classes used in the experiments.

Users working with their own prepared version of GFISHERD24 can update the `path` field in the appropriate configuration file to point to their local dataset.

## Tracking and Evaluation Data

Tracking was run on the GFISHERD24 test image sequences.

For evaluation, a separately prepared MOT-format version of the test data was used. This dataset contains the 45 test sequences used in the experiments, including the image frames used for tracking and the corresponding MOT ground truth.

Each sequence follows a structure such as:

```text
test/
├── sequence_01/
│   ├── img1/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   ├── gt/
│   │   └── gt.txt
│   └── seqinfo.ini
├── sequence_02/
│   └── ...
├── seqmaps/
│   └── test.txt
└── ...
```

Each sequence contains:

- `img1/` with the ordered image frames used for tracking
- `gt/gt.txt` with the MOT ground truth used for evaluation
- `seqinfo.ini` with sequence information

`seqmaps/test.txt` lists the 45 test sequence names used for evaluation.

The MOT ground truth was prepared separately and is not included in this repository.

## Tracking Result Conversion

Tracking predictions produced by Ultralytics are converted to MOT-format result files using:

```text
src/evaluate/convert_to_mot.py
```

This script converts the tracking predictions for comparison with the separately prepared MOT ground truth.

After conversion, small bounding boxes are filtered and the evaluation files are prepared using:

```text
src/evaluate/filter_small_boxes.py
src/evaluate/prepare_evaluation.py
```

The resulting tracking predictions are evaluated against the MOT ground truth using MOTMetrics and TrackEval.
