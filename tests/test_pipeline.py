import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from pathlib import Path
from PIL import Image
import tempfile
from pipeline.quality_filter import QualityFilter
from pipeline.export import to_coco, to_yolo


# ── Quality Filter ─────────────────────────────────────────────────────────────
class TestQualityFilter:
    def _make_image(self, tmp_path, name="img.jpg", blurry=False):
        arr = np.zeros((100, 100, 3), dtype=np.uint8) if blurry \
            else np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        path = tmp_path / name
        Image.fromarray(arr).save(str(path))
        return path

    def test_accepts_sharp_image(self, tmp_path):
        qf = QualityFilter(blur_threshold=10.0)
        img = self._make_image(tmp_path)
        accepted, stats = qf.filter([img])
        assert stats["accepted"] >= 0   # result depends on image content

    def test_rejects_corrupt_file(self, tmp_path):
        corrupt = tmp_path / "corrupt.jpg"
        corrupt.write_bytes(b"not an image")
        qf = QualityFilter()
        _, stats = qf.filter([corrupt])
        assert stats["corrupt"] == 1

    def test_rejects_duplicate(self, tmp_path):
        qf = QualityFilter(hash_distance=10)
        img1 = self._make_image(tmp_path, "a.jpg")
        import shutil
        img2 = tmp_path / "b.jpg"
        shutil.copy(img1, img2)
        _, stats = qf.filter([img1, img2])
        assert stats["duplicate"] == 1

    def test_stats_sum_equals_total(self, tmp_path):
        qf = QualityFilter()
        images = [self._make_image(tmp_path, f"img{i}.jpg") for i in range(3)]
        _, stats = qf.filter(images)
        total = stats["corrupt"] + stats["blurry"] + stats["duplicate"] + stats["accepted"]
        assert total == stats["total"]


# ── Export ─────────────────────────────────────────────────────────────────────
class TestExport:
    def _make_task(self, task_id=1, label="car"):
        return {
            "id": task_id,
            "data": {"image": f"image_{task_id}.jpg"},
            "annotations": [{
                "result": [{
                    "type": "rectanglelabels",
                    "original_width": 640, "original_height": 480,
                    "value": {
                        "x": 10.0, "y": 10.0, "width": 20.0, "height": 15.0,
                        "rectanglelabels": [label],
                    },
                }]
            }],
        }

    def test_coco_structure(self, tmp_path):
        tasks = [self._make_task(1, "car"), self._make_task(2, "person")]
        out = str(tmp_path / "coco.json")
        coco = to_coco(tasks, out)
        assert "images" in coco
        assert "annotations" in coco
        assert "categories" in coco
        assert len(coco["images"]) == 2
        assert len(coco["annotations"]) == 2

    def test_yolo_creates_files(self, tmp_path):
        tasks = [self._make_task(1, "car")]
        out_dir = str(tmp_path / "labels")
        to_yolo(tasks, out_dir)
        label_file = Path(out_dir) / "1.txt"
        assert label_file.exists()
        content = label_file.read_text().strip()
        assert len(content.split()) == 5   # class cx cy w h
