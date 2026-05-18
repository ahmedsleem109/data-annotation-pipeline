"""
Export approved Label Studio annotations to COCO JSON / YOLO format
and upload versioned dataset to S3.
"""
import json
import os
import boto3
from pathlib import Path
from datetime import datetime


def to_coco(annotations: list[dict], output_path: str) -> dict:
    """Convert Label Studio export to COCO JSON format."""
    coco = {"images": [], "annotations": [], "categories": []}
    category_map = {}
    ann_id = 1

    for task in annotations:
        img_id = task["id"]
        file_name = Path(task["data"].get("image", f"image_{img_id}.jpg")).name
        coco["images"].append({"id": img_id, "file_name": file_name})

        for result in task.get("annotations", [{}])[0].get("result", []):
            if result.get("type") != "rectanglelabels":
                continue
            val = result["value"]
            label = val["rectanglelabels"][0]

            if label not in category_map:
                cat_id = len(category_map) + 1
                category_map[label] = cat_id
                coco["categories"].append({"id": cat_id, "name": label})

            orig_w = result["original_width"]
            orig_h = result["original_height"]
            x = val["x"] / 100 * orig_w
            y = val["y"] / 100 * orig_h
            w = val["width"] / 100 * orig_w
            h = val["height"] / 100 * orig_h

            coco["annotations"].append({
                "id": ann_id, "image_id": img_id,
                "category_id": category_map[label],
                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
            })
            ann_id += 1

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"COCO annotations saved: {output_path}")
    return coco


def to_yolo(annotations: list[dict], output_dir: str) -> None:
    """Convert Label Studio export to YOLO .txt format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_set = set()

    for task in annotations:
        img_id = task["id"]
        lines = []
        for result in task.get("annotations", [{}])[0].get("result", []):
            if result.get("type") != "rectanglelabels":
                continue
            val = result["value"]
            label = val["rectanglelabels"][0]
            label_set.add(label)
            cx = (val["x"] + val["width"] / 2) / 100
            cy = (val["y"] + val["height"] / 2) / 100
            w = val["width"] / 100
            h = val["height"] / 100
            label_idx = sorted(label_set).index(label)
            lines.append(f"{label_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        label_file = output_dir / f"{img_id}.txt"
        label_file.write_text("\n".join(lines))

    classes_file = output_dir / "classes.txt"
    classes_file.write_text("\n".join(sorted(label_set)))
    print(f"YOLO labels saved: {output_dir} ({len(annotations)} files)")


def upload_dataset_to_s3(local_dir: str, bucket: str,
                          dataset_name: str, version: str) -> str:
    s3 = boto3.client("s3")
    prefix = f"datasets/{dataset_name}/{version}"
    local_dir = Path(local_dir)
    uploaded = 0

    for path in local_dir.rglob("*"):
        if path.is_file():
            key = f"{prefix}/{path.relative_to(local_dir)}"
            s3.upload_file(str(path), bucket, key)
            uploaded += 1

    manifest = {
        "dataset": dataset_name, "version": version,
        "timestamp": datetime.utcnow().isoformat(),
        "s3_prefix": f"s3://{bucket}/{prefix}",
        "file_count": uploaded,
    }
    s3.put_object(Bucket=bucket, Key=f"{prefix}/manifest.json",
                  Body=json.dumps(manifest, indent=2),
                  ContentType="application/json")

    print(f"Dataset uploaded: s3://{bucket}/{prefix} ({uploaded} files)")
    return f"s3://{bucket}/{prefix}"
