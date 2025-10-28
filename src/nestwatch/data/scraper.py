"""Utilities to fetch training imagery from DuckDuckGo."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

from duckduckgo_search import DDGS

from nestwatch.config import DEFAULT_DATA_DIR


def download_images(query: str, output_dir: Path, max_results: int = 100) -> None:
    """Download images for a specific query into the provided folder."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_files = [
        int(path.stem)
        for path in output_dir.glob("*.jpg")
        if path.stem.isdigit()
    ]
    next_index = (max(existing_files) + 1) if existing_files else 0

    with DDGS() as ddgs:
        for offset, result in enumerate(ddgs.images(query, max_results=max_results)):
            image_url = result.get("image")
            if not image_url:
                continue
            destination = output_dir / f"{next_index + offset}.jpg"
            try:
                urlretrieve(image_url, destination)
                print(f"Downloaded {image_url} -> {destination}")
            except Exception as exc:  # pragma: no cover - best effort download
                print(f"Failed to fetch {image_url}: {exc}")


def scrape_species(species: Iterable[str], root: Path | None = None) -> None:
    """Download datasets for the provided list of species names."""

    root = Path(root or (DEFAULT_DATA_DIR / "birds"))
    for bird in species:
        query = f"{bird} fågel photo"
        download_images(query, root / bird)


def scrape_negative_examples(root: Path | None = None) -> None:
    """Download non-bird imagery for the "none" class."""

    queries = [
        "empty bird feeder",
        "garden no bird",
        "park scenery",
        "forest without animals",
        "backyard photo",
        "tree branches empty",
        "grass background",
        "sky photo no bird",
    ]

    root = Path(root or (DEFAULT_DATA_DIR / "birds" / "none"))
    for query in queries:
        download_images(query, root)


def scrape_default_dataset() -> None:
    """Download a default dataset that mirrors the training configuration."""

    scrape_species(["blåmes", "talgoxe", "koltrast", "gråsparv"])
    scrape_negative_examples()


__all__ = ["download_images", "scrape_species", "scrape_negative_examples", "scrape_default_dataset"]
