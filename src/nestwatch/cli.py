"""Command line interface for NestWatch workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from nestwatch.config import DeviceConfig, StreamConfig, TrainingConfig, VideoTestConfig
from nestwatch.data import scrape_default_dataset, scrape_species
from nestwatch.inference import consume_stream, run_device_inference, run_video_test
from nestwatch.training import export_torchscript, train_model


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=TrainingConfig().dataset_path)
    parser.add_argument("--epochs", type=int, default=TrainingConfig().num_epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig().batch_size)
    parser.add_argument("--lr", type=float, default=TrainingConfig().learning_rate)
    parser.add_argument("--val-split", type=float, default=TrainingConfig().validation_split)
    parser.add_argument("--num-workers", type=int, default=TrainingConfig().num_workers)
    parser.add_argument("--num-classes", type=int, default=TrainingConfig().num_classes)
    parser.add_argument("--output", type=Path, default=TrainingConfig().output_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NestWatch command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the bird classification model")
    _add_training_arguments(train_parser)

    export_parser = subparsers.add_parser("export", help="Export trained weights to TorchScript")
    export_parser.add_argument("state_dict", type=Path, help="Path to the trained state_dict file")
    export_parser.add_argument(
        "--output",
        type=Path,
        default=DeviceConfig().model_path,
        help="Destination path for the TorchScript model",
    )
    export_parser.add_argument("--num-classes", type=int, default=TrainingConfig().num_classes)

    device_parser = subparsers.add_parser("device", help="Run inference on the Raspberry Pi camera")
    device_parser.add_argument("--model", type=Path, default=DeviceConfig().model_path)
    device_parser.add_argument(
        "--confidence", type=float, default=DeviceConfig().confidence_threshold
    )
    device_parser.add_argument(
        "--snapshot-dir", type=Path, default=DeviceConfig().snapshot_dir
    )

    stream_parser = subparsers.add_parser("stream", help="Consume the live MJPEG stream")
    stream_parser.add_argument("--url", type=str, default=StreamConfig().stream_url)
    stream_parser.add_argument("--model", type=Path, default=StreamConfig().model_path)

    video_parser = subparsers.add_parser("video-test", help="Run inference on a recorded video")
    video_parser.add_argument("--video", type=Path, default=VideoTestConfig().video_path)
    video_parser.add_argument("--model", type=Path, default=VideoTestConfig().model_path)

    scrape_parser = subparsers.add_parser("scrape", help="Download images for training")
    scrape_parser.add_argument(
        "--species",
        nargs="*",
        default=None,
        help="Optional list of species names to download. If omitted the default dataset is fetched.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config = TrainingConfig(
            dataset_path=args.dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            validation_split=args.val_split,
            num_workers=args.num_workers,
            num_classes=args.num_classes,
            output_dir=args.output,
        )
        state_dict_path, _ = train_model(config)
        print(f"[INFO] Training finished. State dict stored at {state_dict_path}")

    elif args.command == "export":
        config = TrainingConfig(num_classes=args.num_classes)
        export_torchscript(args.state_dict, args.output, config)

    elif args.command == "device":
        config = DeviceConfig(
            model_path=args.model,
            confidence_threshold=args.confidence,
            snapshot_dir=args.snapshot_dir,
        )
        run_device_inference(config)

    elif args.command == "stream":
        config = StreamConfig(stream_url=args.url, model_path=args.model)
        consume_stream(config)

    elif args.command == "video-test":
        config = VideoTestConfig(video_path=args.video, model_path=args.model)
        run_video_test(config)

    elif args.command == "scrape":
        if args.species:
            scrape_species(args.species)
        else:
            scrape_default_dataset()

    else:  # pragma: no cover - defensive programming
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
