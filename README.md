# Automated Data Annotation and Curation Pipeline

> Python pipeline for automated image dataset collection, quality filtering, pre-annotation with an object detection model, and human-in-the-loop review via Label Studio — with versioned storage on AWS S3.

---

## Overview

Building high-quality labeled datasets is one of the biggest bottlenecks in any ML project. This pipeline automates the tedious parts — collection, deduplication, quality filtering, and pre-annotation — while keeping a human in the loop for final review.

The result is a structured, versioned dataset on S3 that is ready for model training, with a full audit trail of every annotation decision.

---

## Pipeline Stages

```
Raw Image Sources (local / S3 / web)
          │
          ▼
┌─────────────────────┐
│  1. Ingestion        │  ← collect images, normalize formats
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Quality Filter   │  ← remove blurry, duplicate, corrupt images
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Pre-Annotation   │  ← run lightweight detector (YOLOv8-nano)
│     (auto-label)     │     to generate bounding box proposals
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. Human Review     │  ← Label Studio UI for accept/reject/correct
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. Export & Store   │  ← versioned S3 data lake, COCO/YOLO format
└─────────────────────┘
```

---

## Features

- **Automated Ingestion** — Supports local directories, S3 prefixes, and URL lists as sources
- **Quality Filtering** — Detects and removes blurry images (Laplacian variance), near-duplicates (perceptual hash), and corrupt files
- **Auto Pre-Annotation** — Runs YOLOv8-nano to generate bounding box proposals, reducing manual labeling time by ~60%
- **Label Studio Integration** — Uploads pre-annotated tasks to Label Studio for efficient human review
- **Versioned S3 Data Lake** — Exports curated datasets to structured S3 prefixes (`datasets/<name>/v<N>/`)
- **Format Support** — Exports in COCO JSON and YOLO `.txt` formats
- **Reproducibility** — Every dataset version includes a `manifest.json` with source hashes and annotation stats
- **Docker** — Fully containerized with Docker Compose

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10 |
| Object Detection | Ultralytics YOLOv8 |
| Annotation UI | Label Studio |
| Storage | AWS S3, boto3 |
| Image Processing | OpenCV, Pillow, imagehash |
| Containerization | Docker Compose |
| Testing | pytest |

---

## Project Structure

```
data-annotation-pipeline/
├── pipeline/
│   ├── ingest.py              # Image collection and normalization
│   ├── quality_filter.py      # Blur detection, deduplication, corruption check
│   ├── pre_annotate.py        # YOLOv8 inference → Label Studio task format
│   ├── label_studio_client.py # Upload/download tasks via Label Studio API
│   └── export.py              # Convert annotations → COCO / YOLO, upload to S3
├── storage/
│   └── s3_client.py           # S3 upload/download helpers with versioning
├── scripts/
│   ├── run_pipeline.py        # CLI: run full pipeline end-to-end
│   └── export_dataset.py      # CLI: export approved annotations from Label Studio
├── configs/
│   └── pipeline_config.yaml   # Paths, thresholds, S3 bucket, Label Studio URL
├── tests/
│   ├── test_quality_filter.py
│   ├── test_pre_annotate.py
│   └── test_export.py
├── docker-compose.yml         # Label Studio + pipeline services
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- AWS CLI configured
- Label Studio (included via Docker Compose)

### Installation

```bash
git clone https://github.com/ahmedsleem109/data-annotation-pipeline.git
cd data-annotation-pipeline
pip install -r requirements.txt
```

### Start Label Studio

```bash
docker-compose up -d label-studio
# Label Studio will be available at http://localhost:8080
```

### Configuration

Edit `configs/pipeline_config.yaml`:

```yaml
s3:
  bucket: your-bucket
  dataset_name: my-dataset
  version: v1

quality:
  blur_threshold: 100.0      # Laplacian variance threshold
  hash_distance_threshold: 8  # Perceptual hash distance for near-duplicates

label_studio:
  url: http://localhost:8080
  api_key: your-api-key
  project_id: 1

pre_annotation:
  model: yolov8n.pt
  confidence_threshold: 0.3
```

### Run the Full Pipeline

```bash
# Full pipeline: ingest → filter → pre-annotate → upload to Label Studio
python scripts/run_pipeline.py --source ./raw_images --config configs/pipeline_config.yaml

# After human review in Label Studio, export approved annotations
python scripts/export_dataset.py --format coco --config configs/pipeline_config.yaml
```

---

## S3 Data Lake Structure

```
s3://your-bucket/
└── datasets/
    └── my-dataset/
        ├── v1/
        │   ├── images/
        │   ├── annotations/
        │   │   ├── annotations.json    # COCO format
        │   │   └── labels/             # YOLO format
        │   └── manifest.json           # Version metadata + stats
        └── v2/
            └── ...
```

---

## Results

| Metric | Value |
|---|---|
| Pre-annotation acceptance rate | ~72% (no edits needed) |
| Manual labeling time reduction | ~60% vs. labeling from scratch |
| Duplicate/blur removal rate | ~15% of raw images filtered |

---

## Testing

```bash
pytest tests/ -v
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
