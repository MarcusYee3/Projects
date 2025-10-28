import os
import urllib.request

os.makedirs('data/raw', exist_ok=True)
urls = {
    'metadata': 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/.../metadata.csv',
    'images': 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/.../image_data.npy'
}

for name, url in urls.items():
    path = f"data/raw/{name}.{'csv' if 'metadata' in name else 'npy'}"
    urllib.request.urlretrieve(url, path)
    print(f"✅ Downloaded {name} to {path}")