"""
CLI: export approved annotations from Label Studio and upload to S3.
"""
import argparse
import yaml
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.label_studio_client import LabelStudioClient
from pipeline.export import to_coco, to_yolo, upload_dataset_to_s3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["coco", "yolo", "both"],
                        default="both")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--output-dir", default="./exports")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    client = LabelStudioClient(
        url=cfg["label_studio"]["url"],
        api_key=cfg["label_studio"]["api_key"],
        project_id=cfg["label_studio"]["project_id"],
    )

    print("Fetching annotations from Label Studio...")
    annotations = client.get_annotations()
    print(f"  Retrieved {len(annotations)} annotated tasks")

    if args.format in ("coco", "both"):
        to_coco(annotations, f"{args.output_dir}/annotations/coco.json")

    if args.format in ("yolo", "both"):
        to_yolo(annotations, f"{args.output_dir}/annotations/labels")

    s3_cfg = cfg["s3"]
    uri = upload_dataset_to_s3(
        args.output_dir, s3_cfg["bucket"],
        s3_cfg["dataset_name"], s3_cfg["version"],
    )
    print(f"\n✓ Dataset exported and uploaded to: {uri}")


if __name__ == "__main__":
    main()
