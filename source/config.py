import getpass
import json
import os
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


# ---------------------------------------------------------------------------
# Shared LLM endpoint config for the tutorial notebooks
# ---------------------------------------------------------------------------
# Single source of truth: results/kdd26_memdiag_config.json (gitignored, user
# specific). setup_llm() prompts once (Colab Drive or local file) and is then
# reused by every notebook / session, instead of each notebook re-implementing
# the prompt+persist cell. Never prints absolute paths.
_CONFIG_FILENAME = "kdd26_memdiag_config.json"
_DRIVE_CONFIG_PATH = "/content/drive/MyDrive/kdd26_memdiag_config.json"

# Drive mounts at most once per session (the original notebook cell could re-mount).
_MOUNT_DONE = None


def _config_path_with_mount():
    """Resolve the writable config path, mounting Google Drive once in Colab.

    Returns (path, on_drive). Falls back to the local results dir outside Colab
    or when Drive is unavailable. Mirrors the original per-notebook prompt cell.
    """
    global _MOUNT_DONE
    if _MOUNT_DONE is not None:
        return _MOUNT_DONE
    try:
        import google.colab  # noqa: F401  — presence implies a Colab runtime
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        _MOUNT_DONE = (Path(_DRIVE_CONFIG_PATH), True)
    except Exception:
        _MOUNT_DONE = (RESULTS_DIR / _CONFIG_FILENAME, False)
    return _MOUNT_DONE


def load_llm_config():
    """Read-only config lookup. No Drive mount, no prompt.

    Checks the local results file first, then the Colab Drive path (if Drive is
    already mounted). Returns the parsed dict, or {} if nothing is configured.
    """
    for cand in (RESULTS_DIR / _CONFIG_FILENAME, Path(_DRIVE_CONFIG_PATH)):
        try:
            if cand.exists():
                return json.loads(cand.read_text())
        except Exception:
            pass
    return {}


def save_llm_config(base_url, api_key, model):
    """Persist the three values to the resolved config path (mounting Drive in Colab)."""
    path, on_drive = _config_path_with_mount()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"base_url": base_url, "api_key": api_key, "model": model}, indent=2)
    )
    shown = "Google Drive" if on_drive else "<repo>/" + str(path.relative_to(PROJECT_ROOT))
    print("saved ->", shown)


def setup_llm(prompt=True):
    """Configure the OpenAI-compatible endpoint for the tutorial notebooks.

    prompt=True  (Phase 1-3): mount Drive in Colab, then prompt for base URL /
        model / hidden API key on first run (or when KDD_RECONFIG=1), persist
        them, and reuse everywhere. Headless runs fall back to generic defaults
        and stay offline-safe.
    prompt=False (Phase 4): just read whatever Phase 1-3 saved.

    Sets os.environ['OPENAI_BASE_URL' | 'OPENAI_MODEL' | 'OPENAI_API_KEY'] and
    returns (base_url, api_key, model). Never prints absolute paths.
    """
    cfg = {}
    if prompt:
        path, _on_drive = _config_path_with_mount()
        try:
            cfg = json.loads(path.read_text()) if path.exists() else {}
        except Exception:
            cfg = {}
        if (not cfg) or os.environ.get("KDD_RECONFIG", "0") == "1":
            d_base = cfg.get("base_url") or DEFAULT_BASE_URL   # generic, Colab-executable default
            d_model = cfg.get("model") or DEFAULT_MODEL        # never read local env -> no private-path leak
            print("Set up your OpenAI-compatible endpoint (press Enter to keep the default / run offline):")
            try:
                base = input(f"  1. base URL [{d_base}]: ").strip() or d_base
                model = input(f"  2. model    [{d_model}]: ").strip() or d_model
                key = (
                    getpass.getpass("  3. API key (hidden, Enter = none/offline): ").strip()
                    or cfg.get("api_key", "")
                )
                save_llm_config(base, key, model)
                cfg = {"base_url": base, "api_key": key, "model": model}
            except Exception:
                # non-interactive (headless / nbconvert) -> fall back to env / defaults, stay offline-safe
                cfg = cfg or {
                    "base_url": d_base,
                    "api_key": os.environ.get("OPENAI_API_KEY", ""),
                    "model": d_model,
                }
    else:
        cfg = load_llm_config()

    base_url = cfg.get("base_url") or DEFAULT_BASE_URL
    api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    model = cfg.get("model") or DEFAULT_MODEL
    os.environ.update({"OPENAI_BASE_URL": base_url, "OPENAI_MODEL": model})
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    print("base_url =", base_url)
    print("model    =", model)
    print("api_key  =", ("set " + api_key[:6] + "…") if api_key else "NOT set -> offline fallback")
    return base_url, api_key, model
