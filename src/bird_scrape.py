import os
import urllib.request
from duckduckgo_search import DDGS

base_dir = "data/bird-data"

def download_images(query, folder_path, max_results=100):
    os.makedirs(folder_path, exist_ok=True)
    existing_files = os.listdir(folder_path)
    existing_indices = [int(f.split('.')[0]) for f in existing_files if f.endswith('.jpg') and f.split('.')[0].isdigit()]
    next_index = max(existing_indices, default=-1) + 1

    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=max_results)
        for i, result in enumerate(results):
            try:
                url = result['image']
                file_path = os.path.join(folder_path, f"{next_index + i}.jpg")
                urllib.request.urlretrieve(url, file_path)
                print(f"Downloaded: {url}")
            except Exception as e:
                print(f"Failed: {e}")

# === INDIVIDUAL SCRAPER FUNCTIONS ===

def scrape_mes():
    species = ["blåmes", "talgoxe"]
    for bird in species:
        folder = os.path.join(base_dir, "mes", bird)
        query = f"{bird} fågel photo"
        download_images(query, folder, max_results=100)

def scrape_trast():
    species = ["koltrast", "gråsparv"]
    for bird in species:
        folder = os.path.join(base_dir, "trast", bird)
        query = f"{bird} fågel photo"
        download_images(query, folder, max_results=100)

def scrape_none():
    folder = os.path.join(base_dir, "none")
    queries = [
        "empty bird feeder",
        "garden no bird",
        "park scenery",
        "forest without animals",
        "backyard photo",
        "tree branches empty",
        "grass background",
        "sky photo no bird"
    ]
    for q in queries:
        download_images(q, folder, max_results=50)

# === CALL ONE OF THESE TO SCRAPE ===
# scrape_mes()
# scrape_trast()
scrape_none()
