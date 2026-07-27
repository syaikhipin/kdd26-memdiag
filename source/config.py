from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
TOPICS_DIR = DATA_DIR / "topics"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_LOCOMO_PATH = TOPICS_DIR / "locomo" / "locomo10.json"
DEFAULT_AUTORESEARCH_DIR = TOPICS_DIR / "autoresearch"
DEFAULT_USE_CASES = ["locomo", "autoresearch", "memoryarena", "longmemeval"]

DEFAULT_STRATEGIES = [
    "no_memory",
    "verbatim",
    "extracted_facts",
    "episodic",
    "hybrid",
]


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str = "synthetic"
    backend: str = "offline"
    episodes: int = 20
    seed: int = 0
    top_k: int = 5
    target_score: float = 0.82
    tasks_path: Path = DATA_DIR / "synthetic_research_tasks.json"
    results_dir: Path = RESULTS_DIR
    locomo_path: Path = DEFAULT_LOCOMO_PATH
    autoresearch_dir: Path = DEFAULT_AUTORESEARCH_DIR
    max_conversations: int | None = 1
    max_questions: int | None = 10
    use_cases: tuple[str, ...] = tuple(DEFAULT_USE_CASES)
    ideas_per_case: int = 2
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key_env: str = "OPENAI_API_KEY"
    visualize: bool = False
