# NestWatch

NestWatch is a smart bird feeder built around a Raspberry Pi 4 that performs
real-time species recognition and publishes a live video stream. The project
combines a PyTorch-based classifier, a Picamera 2 imaging pipeline, and a
3D-printable enclosure with accompanying build documentation.

## Features

-  **Automated bird recognition** using a fine-tuned MobileNetV2 classifier.
-  **Low-latency live stream** rendered from the Raspberry Pi camera with
  inference overlays.
-  **Hardware documentation and CAD scaffolding** for fabricating the feeder
  enclosure and mounting hardware.
-  **Unified CLI** (`python -m nestwatch.cli`) to train, export, deploy, and
  test the model across environments.

## Repository layout

```
.
├── artifacts/              # Saved models and TorchScript exports
├── data/                   # Datasets and evaluation media
├── docs/                   # Architecture, hardware, and process documentation
├── examples/               # Legacy experiments and demo scripts
├── hardware/               # Placeholder for CAD and wiring assets
├── src/nestwatch/          # Source code packaged as a Python module
└── requirements.txt        # Python dependencies
```

Refer to [`docs/architecture.md`](docs/architecture.md) for a deeper overview of
how the software pieces fit together.

## Getting started

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
# On Raspberry Pi add the camera dependencies
pip install -e .[raspberrypi]
```

### 2. Acquire training data

Use the built-in scraper to seed the dataset folder or copy your own curated
images into `data/birds/<class-name>/` directories.

```bash
python -m nestwatch.cli scrape
```

Each class corresponds to the label order defined in
[`nestwatch/config.py`](src/nestwatch/config.py).

### 3. Train the classifier

```bash
python -m nestwatch.cli train --dataset data/birds --epochs 15
```

Training artifacts (the PyTorch `state_dict`) will be stored in
`artifacts/models/bird_classifier_state_dict.pth`.

### 4. Export to TorchScript

```bash
python -m nestwatch.cli export artifacts/models/bird_classifier_state_dict.pth \
  --output artifacts/models/bird_classifier.pt
```

The resulting `bird_classifier.pt` file runs directly on the Raspberry Pi.

### 5. Deploy on the Raspberry Pi

Copy the repository (or the `src/` package and TorchScript model) to the Pi and
install the dependencies, ensuring `picamera2` support is available. Launch the
headless inference loop:

```bash
python -m nestwatch.cli device --model artifacts/models/bird_classifier.pt
```

Snapshots of confirmed detections will be written to `artifacts/snapshots/`.

### 6. View the live stream

From a workstation on the same network, connect to the MJPEG feed exposed by the
Pi:

```bash
python -m nestwatch.cli stream --url http://<pi-hostname>:5000/video_feed \
  --model artifacts/models/bird_classifier.pt
```

For offline evaluation against stored footage, run:

```bash
python -m nestwatch.cli video-test --video data/test_videos/garden_birds.mp4
```

## Hardware & fabrication

- Assembly instructions: [`docs/hardware/assembly.md`](docs/hardware/assembly.md)
- Bill of materials: [`docs/hardware/bill_of_materials.md`](docs/hardware/bill_of_materials.md)
- CAD and wiring placeholders: [`hardware/`](hardware/)

## Contributing

1. Create a feature branch and make your changes.
2. Ensure new modules follow the package layout under `src/nestwatch/`.
3. Run linting/tests where applicable and open a pull request describing the
   updates.

For questions or suggestions, feel free to open an issue.
