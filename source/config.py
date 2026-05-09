from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DEFAULT_BASE_URL = "http://127.0.0.1:8317/api/provider/codex/v1"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_LOCOMO_PATH = PROJECT_ROOT / "memory-probe" / "data" / "locomo10.json"
DEFAULT_AUTORESEARCH_DIR = PROJECT_ROOT / "autoresearch"
DEFAULT_USE_CASES = ["locomo", "autoresearch", "hpo", "memoryarena", "longmemeval"]

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
