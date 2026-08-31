import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLOv26 detector on GFISHERD24"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to the GFISHERD24 dataset YAML file",
    )

    parser.add_argument(
        "--model",
        default="yolo26l.pt",
        help="Pretrained YOLO checkpoint used to initialize training",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=75,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size",
    )

    parser.add_argument(
        "--project",
        default="runs/train",
        help="Directory for training outputs",
    )

    parser.add_argument(
        "--name",
        default="yolo26l_75e",
        help="Name of the training run",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
