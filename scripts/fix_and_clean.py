#!/usr/bin/env python3
"""One-shot fix script: re-apply the provider refactoring + fix broken files + clean dahono.
Run from experiment/github_submission/"""
import re, sys
from pathlib import Path

ROOT = Path(".")
fixes_applied = 0

def write_file(rel_path, content):
    global fixes_applied
    Path(rel_path).write_text(content)
    fixes_applied += 1
    print(f"  wrote {rel_path}")

def patch_file(rel_path, replacements):
    global fixes_applied
    p = Path(rel_path)
    if not p.exists():
        print(f"  SKIP {rel_path} (not found)")
        return
    content = p.read_text()
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1
        else:
            print(f"  WARN: pattern not found in {rel_path}: {old[:50]}...")
    p.write_text(content)

# ============================================================
# 1. config.py — complete rewrite (paths + clean defaults)
# ============================================================
print("=== 1. config.py ===")
write_file("source/config.py", '''from dataclasses import dataclass
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
''')

# ============================================================
# 2. research_loop.py — provider dispatch (complete rewrite)
# ============================================================
print("=== 2. research_loop.py ===")
write_file("source/research_loop.py", '''"""Benchmark loops over the genuine memory providers."""
from __future__ import annotations
from typing import Any
from corpus_adapters import benchmark_item_records, locomo_records
from data_loaders import Conversation, MemoryBenchmarkItem, iter_locomo_questions
from diagnostics import precision_recall
from providers import build_provider


def _flatten_record_ids(retrieved) -> list[str]:
    ids: list[str] = []
    for item in retrieved:
        for rid in item.record_ids:
            if rid and rid not in ids:
                ids.append(rid)
    return ids


def _failure_category(has_retrieved: bool, hits: int, relevant_count: int) -> str:
    if not has_retrieved:
        return "no_memory_available"
    if hits == 0 and relevant_count:
        return "retrieval_miss"
    if hits < relevant_count:
        return "partial_evidence_retrieved"
    return "none"


def _truncate(text: str, limit: int = 800) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def run_locomo_memory_debug(conversations, strategy_names, top_k, max_conversations, max_questions, embedder, llm):
    if max_conversations is not None:
        conversations = conversations[:max_conversations]
    all_records, memories = [], {}
    for sn in strategy_names:
        provider = build_provider(sn, embedder, llm)
        gs = 0
        for conv in conversations:
            provider.ingest(locomo_records(conv))
            for qi, qa in iter_locomo_questions(conv, max_questions):
                gs += 1
                ret, lat = provider.retrieve(qa.question, top_k=top_k)
                rids = _flatten_record_ids(ret)
                diag = precision_recall(rids, qa.evidence_ids)
                fail = _failure_category(bool(ret), int(diag["hits"]), len(qa.evidence_ids))
                all_records.append({
                    "dataset": "LoCoMo", "strategy": sn, "global_step": gs, "conv_id": conv.conv_id,
                    "question_idx": qi, "question": qa.question, "gold_answer": qa.answer, "category": qa.category,
                    "evidence_ids": qa.evidence_ids, "retrieved_memory_ids": rids, "retrieved_record_ids": list(rids),
                    "retrieved_texts": [_truncate(i.content) for i in ret], "retrieval_scores": [i.score for i in ret],
                    "retrieval_precision": diag["precision"], "retrieval_recall": diag["recall"], "evidence_hit": diag["evidence_hit"],
                    "failure_category": fail, "diagnosed_failure": fail, "answer_mode": "provider_retrieval",
                    "answer": "retrieved_evidence_answerable" if diag["evidence_hit"] else "insufficient_retrieved_evidence",
                    "memory_utilized": bool(diag["evidence_hit"]), "utilization_category": "provisional",
                    "latency_ms": lat, "cost_units": 0.1 + 0.02 * len(ret) + 0.001 * provider.size(),
                    "memory_entries": provider.size(), "provider_export": provider.export(),
                })
        memories[sn] = provider.export()
    return all_records, memories


def run_memory_benchmark(items, strategy_names, top_k, embedder, llm):
    all_records, memories = [], {}
    for sn in strategy_names:
        gs = 0
        for item in items:
            provider = build_provider(sn, embedder, llm)
            provider.ingest(benchmark_item_records(item))
            gs += 1
            ret, lat = provider.retrieve(item.question, top_k=top_k)
            rids = _flatten_record_ids(ret)
            diag = precision_recall(rids, item.relevant_ids)
            fail = _failure_category(bool(ret), int(diag["hits"]), len(item.relevant_ids))
            all_records.append({
                "dataset": item.dataset, "strategy": sn, "global_step": gs, "item_id": item.item_id,
                "question": item.question, "gold_answer": item.answer, "category": item.category,
                "evidence_ids": item.relevant_ids, "retrieved_memory_ids": rids, "retrieved_record_ids": list(rids),
                "retrieved_texts": [_truncate(i.content) for i in ret], "retrieval_scores": [i.score for i in ret],
                "retrieval_precision": diag["precision"], "retrieval_recall": diag["recall"], "evidence_hit": diag["evidence_hit"],
                "failure_category": fail, "diagnosed_failure": fail, "answer_mode": "provider_retrieval",
                "answer": "retrieved_evidence_answerable" if diag["evidence_hit"] else "insufficient_retrieved_evidence",
                "memory_utilized": bool(diag["evidence_hit"]), "utilization_category": "provisional",
                "latency_ms": lat, "cost_units": 0.1 + 0.02 * len(ret) + 0.001 * provider.size(),
                "memory_entries": provider.size(), "provider_export": provider.export(),
            })
        memories[sn] = {"strategy": sn, "items": len(items)}
    return all_records, memories
''')

# ============================================================
# 3. data_loaders.py — MemoryArena fix + topics path
# ============================================================
print("=== 3. data_loaders.py patches ===")
patch_file("source/data_loaders.py", [
    # Path
    ('REAL_DATA_DIR = Path(__file__).resolve().parent / "data" / "real"',
     'REAL_DATA_DIR = Path(__file__).resolve().parent / "data" / "topics"'),
    # MemoryArena relevance fix
    ('''def _memoryarena_relevant_ids(records: list[dict[str, Any]], question_idx: int) -> list[str]:
    indexed = [record for record in records if record["metadata"].get("background_idx") == question_idx]
    if indexed:
        return [record["id"] for record in indexed]
    return [record["id"] for record in records]''',
     '''_MA_STOP = {"the","a","an","and","or","of","to","in","on","for","is","are","was","were","be","been","being","that","this","these","those","it","its","as","at","by","with","from","into","has","have","had","not","no","yes","do","does","did","what","which","who","whom","whose","when","where","why","how","can","could"}

def _ma_tok(text):
    return re.findall(r"[a-zA-Z0-9_]+", _stringify(text).lower())

def _memoryarena_relevant_ids(records: list[dict[str, Any]], answer: Any) -> list[str]:
    atoks = {t for t in _ma_tok(answer) if t not in _MA_STOP and len(t) > 2}
    if not atoks:
        return []
    relevant = []
    for record in records:
        rtoks = set(_ma_tok(str(record.get("content", ""))))
        overlap = len(atoks & rtoks)
        if overlap and overlap / len(atoks) >= 0.15:
            relevant.append(record["id"])
    return relevant'''),
    # Update the caller
    ('relevant = _memoryarena_relevant_ids(base_records, question_idx)',
     'relevant = _memoryarena_relevant_ids(base_records, answer)'),
])

# ============================================================
# 4. evaluators.py — non-circular metrics
# ============================================================
print("=== 4. evaluators.py patch ===")
patch_file("source/evaluators.py", [
    ('''        answer_correctness = _overlap(gold, answer)
        if answer in {"retrieved_evidence_answerable", "insufficient_retrieved_evidence"}:
            answer_correctness = float(bool(record.get("evidence_hit") or record.get("memory_utilized")))
        context_relevance = max(_overlap(question, context_text), float(record.get("retrieval_recall", 0.0)))
        if answer in {"retrieved_evidence_answerable", "insufficient_retrieved_evidence"}:
            faithfulness = float(bool(record.get("evidence_hit") and record.get("memory_utilized")))
        else:
            faithfulness = _overlap(answer, context_text) if answer else float(bool(record.get("memory_utilized")))
        semantic = round((answer_correctness + context_relevance + faithfulness) / 3.0, 4)
        passed = semantic >= 0.5 or bool(record.get("evidence_hit"))''',
     '''        answer_correctness = _overlap(gold, context_text)
        context_relevance = _overlap(question, context_text)
        faithfulness = _set_overlap(set(tokenize(gold)), set(tokenize(context_text)))
        semantic = round((answer_correctness + context_relevance + faithfulness) / 3.0, 4)
        passed = bool(contexts) and semantic >= 0.5'''),
])

# ============================================================
# 5. run.py — embedder wiring + utilization + use-cases
# ============================================================
print("=== 5. run.py patches ===")
patch_file("source/run.py", [
    # import make_embedder
    ("from evaluators import make_evaluators, redact_secret\nfrom idea_generator import generate_ideas",
     "from evaluators import make_evaluators, redact_secret\nfrom embedder import make_embedder\nfrom idea_generator import generate_ideas"),
    # use-cases choices
    ('choices=["locomo", "autoresearch", "hpo", "memoryarena", "longmemeval", "lcbench"]',
     'choices=["locomo", "autoresearch", "memoryarena", "longmemeval"]'),
    # Add _recompute_utilization + _make_embedder_and_llm before run_locomo
    ("def run_locomo(args: argparse.Namespace) -> tuple[dict, dict]:",
     '''def _recompute_utilization(records):
    for record in records:
        gold = set(tokenize(str(record.get("gold_answer", ""))))
        context_tokens = set()
        for text in record.get("retrieved_texts", []):
            context_tokens.update(tokenize(str(text)))
        retrieved = bool(record.get("retrieved_texts"))
        used = bool(gold & context_tokens) if gold else retrieved
        record["memory_utilized"] = used
        if not retrieved:
            record["utilization_category"] = "no_memory_available"
        elif used:
            record["utilization_category"] = "context_mentions_answer"
        else:
            record["utilization_category"] = "retrieved_but_answer_absent"

def _make_embedder_and_llm(args):
    from llm_client import LLMConfig, make_client as _mc
    llm = _mc(LLMConfig(backend=args.backend, base_url=args.base_url, model=args.model, api_key_env=args.api_key_env, api_key=args.api_key))
    embedder = make_embedder(backend=args.backend, base_url=args.base_url, api_key=args.api_key or os.environ.get(args.api_key_env), model="text-embedding-3-small")
    return embedder, llm

def run_locomo(args: argparse.Namespace) -> tuple[dict, dict]:'''),
    # Wire embedder+llm into run_locomo
    ("    conversations = load_locomo(args.locomo_path)\n    records, memories = run_locomo_memory_debug(",
     "    conversations = load_locomo(args.locomo_path)\n    embedder, llm = _make_embedder_and_llm(args)\n    records, memories = run_locomo_memory_debug("),
    ("        answer_mode=\"offline_evidence_heuristic\" if args.backend == \"offline\" else \"llm_with_retrieved_memory\",\n    )\n    _apply_evaluators(records, args)\n    summary = summarize_locomo",
     "        embedder=embedder,\n        llm=llm,\n    )\n    _recompute_utilization(records)\n    _apply_evaluators(records, args)\n    summary = summarize_locomo"),
    # Wire into run_real
    ("    dataset_items = _load_real_items(args)\n    for dataset_name, items in dataset_items.items():\n        dataset_records, dataset_memories = run_memory_benchmark(",
     "    dataset_items = _load_real_items(args)\n    embedder, llm = _make_embedder_and_llm(args)\n    for dataset_name, items in dataset_items.items():\n        dataset_records, dataset_memories = run_memory_benchmark("),
    ("            answer_mode=\"offline_evidence_heuristic\" if args.backend == \"offline\" else \"llm_with_retrieved_memory\",\n        )\n        records.extend",
     "            embedder=embedder,\n            llm=llm,\n        )\n        records.extend"),
    # Add _recompute + utilization lab in run_real after summary
    ("    summary = summarize_real_datasets(records)\n    payload",
     "    summary = summarize_real_datasets(records)\n    from probes.utilization_probe import build_utilization_fixture, run_utilization_lab\n    fixture = build_utilization_fixture(args.locomo_path, args.longmemeval_dir, args.longmemeval_files, per_dataset=20, seed=args.seed)\n    utilization_lab = run_utilization_lab(fixture, args.strategies, embedder, llm, top_k=args.top_k)\n    summary[\"utilization_lab\"] = utilization_lab\n    _recompute_utilization(records)\n    payload"),
    # Wire into run_tutorial
    ("    conversations = load_locomo(args.locomo_path)\n    locomo_records, memories = run_locomo_memory_debug(",
     "    conversations = load_locomo(args.locomo_path)\n    embedder, llm = _make_embedder_and_llm(args)\n    locomo_records, memories = run_locomo_memory_debug("),
    ("        answer_mode=\"offline_evidence_heuristic\" if args.backend == \"offline\" else \"llm_with_retrieved_memory\",\n    )\n    ideas = generate_ideas",
     "        embedder=embedder,\n        llm=llm,\n    )\n    _recompute_utilization(locomo_records)\n    ideas = generate_ideas"),
    # Tutorial report prose
    ("- LoCoMo is used directly from the local PDF-listed dataset path.\n- LCBench, MemoryArena, HPOBench, and LongMemEval are included in the dataset registry",
     "- LoCoMo is used from the bundled topic data.\n- Autoresearch (real L4 trace), MemoryArena, and LongMemEval are bundled topics"),
])

# ============================================================
# 6. Clean any remaining dahono in source
# ============================================================
print("=== 6. Clean dahono from source ===")
for path in ROOT.rglob("*.py"):
    if "__pycache__" in str(path):
        continue
    content = path.read_text()
    for old, new in [("https://api.openai.com/v1", "https://api.openai.com/v1"),
                     ("api.openai.com", "api.openai.com"),
                     ("gpt-4o", "gpt-4o")]:
        content = content.replace(old, new)
    if content != path.read_text():
        path.write_text(content)

# Also clean data_loaders LCBench/HPOBench stubs
print("=== 7. Clean data_loaders registry ===")
patch_file("source/data_loaders.py", [
    ('DEFAULT_LCBENCH_DIR = REAL_DATA_DIR / "lcbench"', '# LCBench/HPOBench removed (see autoresearch topic)'),
    ('DEFAULT_HPOBENCH_DIR = REAL_DATA_DIR / "hpobench"', ''),
])

print(f"\n=== TOTAL FIXES APPLIED: {fixes_applied} ===")
