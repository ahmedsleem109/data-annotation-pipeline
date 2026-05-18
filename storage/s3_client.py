import boto3
from pathlib import Path


class S3Client:
    def __init__(self, bucket: str):
        self.bucket = bucket
        self.s3 = boto3.client("s3")

    def upload(self, local_path: str, key: str) -> str:
        self.s3.upload_file(local_path, self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def download(self, key: str, local_path: str) -> None:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, key, local_path)

    def list_keys(self, prefix: str) -> list[str]:
        paginator = self.s3.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def upload_directory(self, local_dir: str, prefix: str) -> int:
        local_dir = Path(local_dir)
        count = 0
        for path in local_dir.rglob("*"):
            if path.is_file():
                key = f"{prefix}/{path.relative_to(local_dir)}"
                self.upload(str(path), key)
                count += 1
        return count
