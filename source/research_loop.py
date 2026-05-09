from typing import Any

from data_loaders import Conversation, MemoryBenchmarkItem, iter_locomo_questions, locomo_memory_records
from diagnostics import locomo_retrieval_diagnostics, locomo_utilization_category, precision_recall
from memory_store import MemoryStore
from strategies import ALL_STRATEGIES


def run_locomo_memory_debug(
    conversations: list[Conversation],
    strategy_names: list[str],
    top_k: int,
    max_conversations: int | None,
    max_questions: int | None,
    answer_mode: str = "offline_evidence_heuristic",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_conversations is not None:
        conversations = conversations[:max_conversations]
    all_records = []
    memories = {}
    for strategy_name in strategy_names:
        store = MemoryStore(strategy_name)
        global_step = 0
        for conversation in conversations:
            _ingest_locomo_conversation(store, strategy_name, conversation)
            for question_idx, qa in iter_locomo_questions(conversation, max_questions):
                global_step += 1
                retrieved, latency_ms = store.retrieve(qa.question, top_k=top_k)
                diag = locomo_retrieval_diagnostics(retrieved, qa.evidence_ids)
                answer = "retrieved_evidence_answerable" if diag["evidence_hit"] else "insufficient_retrieved_evidence"
                record = {
                    "dataset": "LoCoMo",
                    "strategy": strategy_name,
                    "global_step": global_step,
                    "conv_id": conversation.conv_id,
                    "question_idx": question_idx,
                    "question": qa.question,
                    "gold_answer": qa.answer,
                    "category": qa.category,
                    "evidence_ids": qa.evidence_ids,
                    "retrieved_memory_ids": [item["entry"].id for item in retrieved],
                    "retrieved_dia_ids": [str(item["entry"].metadata.get("dia_id", "")) for item in retrieved],
                    "retrieved_texts": [item["entry"].content for item in retrieved],
                    "retrieval_scores": [item["score"] for item in retrieved],
                    "retrieval_precision": diag["precision"],
                    "retrieval_recall": diag["recall"],
                    "evidence_hit": diag["evidence_hit"],
                    "failure_category": diag["failure_category"],
                    "diagnosed_failure": diag["failure_category"],
                    "answer_mode": answer_mode,
                    "answer": answer,
                    "memory_utilized": diag["evidence_hit"],
                    "utilization_category": locomo_utilization_category(bool(diag["evidence_hit"]), answer_mode),
                    "latency_ms": latency_ms,
                    "cost_units": 0.1 + 0.02 * len(retrieved) + 0.001 * store.size(),
                    "memory_entries": store.size(),
                }
                all_records.append(record)
        memories[strategy_name] = store.to_dict()
    return all_records, memories


def run_memory_benchmark(
    items: list[MemoryBenchmarkItem],
    strategy_names: list[str],
    top_k: int,
    answer_mode: str = "offline_evidence_heuristic",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_records = []
    memories = {}
    for strategy_name in strategy_names:
        global_step = 0
        for item in items:
            store = MemoryStore(strategy_name)
            _ingest_memory_records(store, strategy_name, item.memory_records, item.item_id)
            global_step += 1
            retrieved, latency_ms = store.retrieve(item.question, top_k=top_k)
            retrieved_record_ids = [str(entry["entry"].metadata.get("record_id", entry["entry"].id)) for entry in retrieved]
            diag = precision_recall(retrieved_record_ids, item.relevant_ids)
            record = {
                "dataset": item.dataset,
                "strategy": strategy_name,
                "global_step": global_step,
                "item_id": item.item_id,
                "question": item.question,
                "gold_answer": item.answer,
                "category": item.category,
                "evidence_ids": item.relevant_ids,
                "retrieved_memory_ids": [entry["entry"].id for entry in retrieved],
                "retrieved_record_ids": retrieved_record_ids,
                "retrieved_texts": [_truncate_text(entry["entry"].content) for entry in retrieved],
                "retrieval_scores": [entry["score"] for entry in retrieved],
                "retrieval_precision": diag["precision"],
                "retrieval_recall": diag["recall"],
                "evidence_hit": diag["evidence_hit"],
                "failure_category": _failure_category(bool(retrieved), diag["hits"], len(item.relevant_ids)),
                "diagnosed_failure": _failure_category(bool(retrieved), diag["hits"], len(item.relevant_ids)),
                "answer_mode": answer_mode,
                "answer": "retrieved_evidence_answerable" if diag["evidence_hit"] else "insufficient_retrieved_evidence",
                "memory_utilized": diag["evidence_hit"],
                "utilization_category": locomo_utilization_category(bool(diag["evidence_hit"]), answer_mode),
                "latency_ms": latency_ms,
                "cost_units": 0.1 + 0.02 * len(retrieved) + 0.001 * store.size(),
                "memory_entries": store.size(),
            }
            all_records.append(record)
        memories[strategy_name] = {"strategy": strategy_name, "items": len(items)}
    return all_records, memories


def _ingest_locomo_conversation(store: MemoryStore, strategy_name: str, conversation: Conversation) -> None:
    if strategy_name == "no_memory":
        return
    records = locomo_memory_records(conversation)
    _ingest_memory_records(store, strategy_name, records, conversation.conv_id)


def _ingest_memory_records(store: MemoryStore, strategy_name: str, records: list[dict[str, Any]], source_task: str) -> None:
    if strategy_name == "no_memory":
        return
    for record in records:
        content = _strategy_content(strategy_name, record)
        metadata = record["metadata"]
        store.add(
            content=content,
            metadata=metadata,
            source_episode=str(metadata.get("session_id") or metadata.get("source_id") or source_task),
            source_task=source_task,
            entry_type=metadata["entry_type"],
        )


def _failure_category(has_retrieved: bool, hits: int, relevant_count: int) -> str:
    if not has_retrieved:
        return "no_memory_available"
    if hits == 0 and relevant_count:
        return "retrieval_miss"
    if hits < relevant_count:
        return "partial_evidence_retrieved"
    return "none"


def _truncate_text(text: str, limit: int = 800) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _strategy_content(strategy_name: str, record: dict[str, Any]) -> str:
    metadata = record["metadata"]
    content = record["content"]
    session_id = metadata.get("session_id") or metadata.get("source_id") or metadata.get("record_id", "unknown")
    record_id = metadata.get("record_id") or metadata.get("dia_id") or session_id
    speaker = metadata.get("speaker") or metadata.get("task_name") or metadata.get("dataset", "source")
    if strategy_name == "no_memory":
        return ""
    if strategy_name == "extracted_facts":
        return f"Fact from {speaker} in {session_id}: {content}"
    if strategy_name == "episodic":
        return f"Episode {session_id} around {metadata.get('timestamp')}: {content}"
    if strategy_name == "hybrid":
        return f"{metadata.get('dataset', 'memory')} evidence {record_id} {session_id} {speaker}: {content}"
    return f"{session_id} {record_id} {content}"
