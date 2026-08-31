# Model Weights

The trained detector weights used in the paper are not included in this repository.

Two YOLOv26-L detectors were trained for the experiments:

- Single-class, with all fish assigned to one `Fish` class
- Multi-class, using 155 fish classes

Both detectors were trained for 75 epochs starting from the Ultralytics `yolo26l.pt` pretrained checkpoint. The `best.pt` checkpoint from each training run was used for tracking and evaluation.

## Pretrained Weights

The Ultralytics `yolo26l.pt` pretrained checkpoint is publicly available through Ultralytics. The training script downloads it automatically if it is not already available.

See the [Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/) for information about the official pretrained models.

The Ultralytics pretrained weights are not included in this repository.

## Trained Detector Weights

Users can train single- and multi-class detector checkpoints using the training instructions in the main repository README.

After training, the detector checkpoints can be organized as:

```text
weights/
├── single_class/
│   └── best.pt
└── multi_class/
    └── best.pt
```

The checkpoint paths are provided to the tracking script using the `--model` argument.

### Single-Class Model

```bash
python src/track/track_dataset.py \
  --model weights/single_class/best.pt \
  --data-root /path/to/mot/test \
  --output runs/tracking_single
```

### Multi-Class Model

```bash
python src/track/track_dataset.py \
  --model weights/multi_class/best.pt \
  --data-root /path/to/mot/test \
  --output runs/tracking_multi
```
