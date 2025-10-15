import imagehash
from PIL import Image

class Deduplicator:
    """
    Remove near-duplicate images using perceptual hashing (pHash)
    """
    def __init__(self, threshold=8):
        self.threshold = threshold

    def dedup(self, images):
        """
        images: list of dicts with 'path' key
        returns: deduplicated list
        """
        hashes = {}
        keep = []
        for img in images:
            try:
                pil = Image.open(img['path']).convert('RGB')
                h = imagehash.phash(pil)
                duplicate = False
                for existing_h in hashes.values():
                    if h - existing_h <= self.threshold:
                        duplicate = True
                        break
                if not duplicate:
                    keep.append(img)
                    hashes[img['path']] = h
            except Exception as e:
                continue
        return keep