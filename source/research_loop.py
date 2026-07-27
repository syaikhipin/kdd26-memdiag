"""Benchmark loops over the genuine memory providers."""
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
