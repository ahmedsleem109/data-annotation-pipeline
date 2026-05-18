from pathlib import Path
from ultralytics import YOLO
import json


class PreAnnotator:
    """
    Runs YOLOv8 inference on images to generate bounding box proposals
    in Label Studio task format for human review.
    """

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.3):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def annotate(self, image_path: Path) -> dict:
        """
        Run inference on a single image and return a Label Studio task dict.

        Args:
            image_path: Path to the image file.

        Returns:
            Label Studio task dict with pre-annotations.
        """
        results = self.model(str(image_path), conf=self.confidence, verbose=False)
        result = results[0]

        predictions = []
        img_w, img_h = result.orig_shape[1], result.orig_shape[0]

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = self.model.names[int(box.cls)]
            score = float(box.conf)

            predictions.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": x1 / img_w * 100,
                    "y": y1 / img_h * 100,
                    "width": (x2 - x1) / img_w * 100,
                    "height": (y2 - y1) / img_h * 100,
                    "rectanglelabels": [label],
                },
                "score": score,
            })

        return {
            "data": {"image": str(image_path)},
            "predictions": [{"result": predictions, "score": max((p["score"] for p in predictions), default=0.0)}],
        }

    def annotate_batch(self, image_paths: list[Path]) -> list[dict]:
        return [self.annotate(p) for p in image_paths]
