import os
import csv
from PIL import Image

def load_images_from_folder(folder_path, exts=('.jpg','.jpeg','.png')):
    """
    Load all image paths from a folder with supported extensions.
    """
    images = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(exts):
            images.append(os.path.join(folder_path, fname))
    return sorted(images)

def save_csv(output_path, rows, fieldnames):
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k,'') for k in fieldnames})

def load_image_pil(path):
    """
    Load PIL image, handle exceptions.
    """
    from PIL import Image
    try:
        return Image.open(path).convert('RGB')
    except Exception as e:
        raise IOError(f"Failed to load image {path}: {e}")