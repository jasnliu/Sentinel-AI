import os
import requests
from PIL import Image
from io import BytesIO
from duckduckgo_search import DDGS

# Create a folder to save the images
save_dir = 'forrest'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Function to search for image URLs
def search_images(keywords, max_images=30):
    print(f"Searching for '{keywords}'")
    results = DDGS().images(keywords, max_results=max_images)
    return [result['image'] for result in results]

# Function to download images
def download_images(image_urls):
    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url,verify=False)
            img = Image.open(BytesIO(response.content))
            img_format = img.format.lower()

            img.save(os.path.join(save_dir, f"forrest_{i+1}.{img_format}"))
            print(f"Downloaded image {i+1} from {url}")
        except Exception as e:
            print(f"Could not download image {i+1}: {e}")

# Search for banana images and download them
search_term = 'forest at a distance'
max_images = 1000
image_urls = search_images(search_term, max_images)
download_images(image_urls)

print(f"Downloaded {len(image_urls)} banana images to '{save_dir}' folder.")

