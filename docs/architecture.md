# System Architecture

The NestWatch platform is composed of three pillars: data, software, and
hardware. The diagram below summarises the workflow.

1. **Data acquisition**
   - `python -m nestwatch.cli scrape` fetches images for each supported species using the
     DuckDuckGo image API.
   - Curated datasets are stored under `data/birds/` with a directory per class.
2. **Model training**
   - `python -m nestwatch.cli train` wraps the training pipeline built around a MobileNetV2
     classifier. Training artifacts are written to `artifacts/models/`.
   - `python -m nestwatch.cli export` converts the trained weights into a TorchScript module
     ready for the Raspberry Pi.
3. **Edge deployment**
   - The Raspberry Pi runs `python -m nestwatch.cli device` which loads the TorchScript module
     and performs inference directly on the Pi Camera using Picamera2.
   - The device hosts an MJPEG stream that can be viewed remotely with
     `python -m nestwatch.cli stream`.

Additional tooling is available under `python -m nestwatch.cli video-test` for local
video analysis and within `examples/` for early experiments.

The training and inference configuration is centralised in
`nestwatch/config.py` to keep reproducibility and deployment aligned.
