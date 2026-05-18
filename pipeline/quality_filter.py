import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import imagehash


class QualityFilter:
    """
    Filters out low-quality images from a dataset:
    - Blurry images (Laplacian variance below threshold)
    - Near-duplicate images (perceptual hash distance)
    - Corrupt/unreadable files
    """

    def __init__(self, blur_threshold: float = 100.0, hash_distance: int = 8):
        self.blur_threshold = blur_threshold
        self.hash_distance = hash_distance
        self._seen_hashes: list = []

    def is_blurry(self, image_path: Path) -> bool:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        return laplacian_var < self.blur_threshold

    def is_corrupt(self, image_path: Path) -> bool:
        try:
            with Image.open(image_path) as img:
                img.verify()
            return False
        except Exception:
            return True

    def is_duplicate(self, image_path: Path) -> bool:
        try:
            with Image.open(image_path) as img:
                h = imagehash.phash(img)
        except Exception:
            return False

        for seen in self._seen_hashes:
            if abs(h - seen) <= self.hash_distance:
                return True

        self._seen_hashes.append(h)
        return False

    def filter(self, image_paths: list[Path]) -> tuple[list[Path], dict]:
        """
        Filter a list of image paths.

        Returns:
            (accepted, stats) where stats contains counts of each rejection reason.
        """
        accepted = []
        stats = {"total": len(image_paths), "corrupt": 0, "blurry": 0, "duplicate": 0, "accepted": 0}

        for path in image_paths:
            if self.is_corrupt(path):
                stats["corrupt"] += 1
                continue
            if self.is_blurry(path):
                stats["blurry"] += 1
                continue
            if self.is_duplicate(path):
                stats["duplicate"] += 1
                continue
            accepted.append(path)
            stats["accepted"] += 1

        return accepted, stats
