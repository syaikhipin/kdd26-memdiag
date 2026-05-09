import json
from pathlib import Path

from models import ResearchTask


def load_tasks(path: Path) -> list[ResearchTask]:
    with open(path, "r", encoding="utf-8") as f:
        raw_tasks = json.load(f)
    return [ResearchTask(**item) for item in raw_tasks]


def all_actions(task: ResearchTask) -> list[str]:
    return list(task.useful_actions) + list(task.failure_actions)


def action_text(task: ResearchTask, action: str) -> str:
    keywords = ", ".join(task.action_keywords.get(action, []))
    return f"{task.domain}: {action}. Keywords: {keywords}. Constraints: {', '.join(task.constraints)}"
