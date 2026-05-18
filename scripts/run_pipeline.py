"""
CLI: run full annotation pipeline end-to-end.
Ingest → Quality Filter → Pre-Annotate → Upload to Label Studio
"""
import argparse
import yaml
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.ingest import ImageIngestor
from pipeline.quality_filter import QualityFilter
from pipeline.pre_annotate import PreAnnotator
from pipeline.label_studio_client import LabelStudioClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Local image directory")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    staging_dir = "./staging/raw"
    print("\n── Stage 1: Ingestion ──────────────────────────")
    ingestor = ImageIngestor(staging_dir)
    images = ingestor.from_local(args.source)

    print("\n── Stage 2: Quality Filtering ──────────────────")
    qf = QualityFilter(
        blur_threshold=cfg["quality"]["blur_threshold"],
        hash_distance=cfg["quality"]["hash_distance_threshold"],
    )
    accepted, stats = qf.filter(images)
    print(f"  Total:     {stats['total']}")
    print(f"  Corrupt:   {stats['corrupt']}")
    print(f"  Blurry:    {stats['blurry']}")
    print(f"  Duplicate: {stats['duplicate']}")
    print(f"  Accepted:  {stats['accepted']}")

    print("\n── Stage 3: Pre-Annotation ─────────────────────")
    annotator = PreAnnotator(
        model_path=cfg["pre_annotation"]["model"],
        confidence=cfg["pre_annotation"]["confidence_threshold"],
    )
    tasks = annotator.annotate_batch(accepted)
    print(f"  Generated {len(tasks)} pre-annotated tasks")

    print("\n── Stage 4: Upload to Label Studio ─────────────")
    client = LabelStudioClient(
        url=cfg["label_studio"]["url"],
        api_key=cfg["label_studio"]["api_key"],
        project_id=cfg["label_studio"]["project_id"],
    )
    task_ids = client.upload_tasks(tasks)
    print(f"  Uploaded {len(task_ids)} tasks")

    stats_ls = client.get_project_stats()
    print(f"\n  Project stats: {stats_ls}")
    print("\n✓ Pipeline complete. Review annotations in Label Studio.")


if __name__ == "__main__":
    main()
