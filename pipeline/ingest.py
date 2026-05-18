"""
Image ingestion: collect images from local dirs, S3 prefixes, or URL lists.
Normalizes all images to JPEG/PNG and saves to a staging directory.
"""
import os
import boto3
import requests
from pathlib import Path
from PIL import Image


class ImageIngestor:
    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def from_local(self, source_dir: str) -> list[Path]:
        source = Path(source_dir)
        collected = []
        for ext in self.SUPPORTED_EXTS:
            for p in source.rglob(f"*{ext}"):
                dest = self.output_dir / p.name
                img = Image.open(p).convert("RGB")
                img.save(dest, "JPEG", quality=95)
                collected.append(dest)
        print(f"Ingested {len(collected)} images from {source_dir}")
        return collected

    def from_s3(self, bucket: str, prefix: str) -> list[Path]:
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        collected = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                ext = Path(key).suffix.lower()
                if ext not in self.SUPPORTED_EXTS:
                    continue
                local_path = self.output_dir / Path(key).name
                s3.download_file(bucket, key, str(local_path))
                collected.append(local_path)
        print(f"Ingested {len(collected)} images from s3://{bucket}/{prefix}")
        return collected

    def from_url_list(self, url_file: str) -> list[Path]:
        collected = []
        with open(url_file) as f:
            urls = [line.strip() for line in f if line.strip()]
        for i, url in enumerate(urls):
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                ext = Path(url).suffix or ".jpg"
                dest = self.output_dir / f"image_{i:06d}{ext}"
                dest.write_bytes(resp.content)
                collected.append(dest)
            except Exception as e:
                print(f"  SKIP {url}: {e}")
        print(f"Ingested {len(collected)}/{len(urls)} images from URLs")
        return collected
