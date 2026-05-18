"""
Label Studio API client.
Uploads pre-annotated tasks and downloads approved annotations.
"""
import requests
import json
from pathlib import Path


class LabelStudioClient:
    def __init__(self, url: str, api_key: str, project_id: int):
        self.base = url.rstrip("/")
        self.headers = {"Authorization": f"Token {api_key}",
                        "Content-Type": "application/json"}
        self.project_id = project_id

    def upload_tasks(self, tasks: list[dict]) -> list[int]:
        """Upload pre-annotated tasks to Label Studio. Returns task IDs."""
        resp = requests.post(
            f"{self.base}/api/projects/{self.project_id}/import",
            headers=self.headers,
            data=json.dumps(tasks),
        )
        resp.raise_for_status()
        result = resp.json()
        print(f"Uploaded {result.get('task_count', 0)} tasks")
        return result.get("task_ids", [])

    def get_annotations(self) -> list[dict]:
        """Download all completed annotations from the project."""
        resp = requests.get(
            f"{self.base}/api/projects/{self.project_id}/export?exportType=JSON",
            headers=self.headers,
        )
        resp.raise_for_status()
        return resp.json()

    def get_project_stats(self) -> dict:
        resp = requests.get(
            f"{self.base}/api/projects/{self.project_id}",
            headers=self.headers,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "total_tasks": data.get("task_count", 0),
            "completed": data.get("num_tasks_with_annotations", 0),
        }

    def delete_all_tasks(self) -> None:
        resp = requests.delete(
            f"{self.base}/api/projects/{self.project_id}/tasks",
            headers=self.headers,
        )
        resp.raise_for_status()
        print("All tasks deleted")
