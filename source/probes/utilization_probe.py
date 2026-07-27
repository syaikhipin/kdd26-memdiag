"""Utilization probes: legacy helpers + 3-arm contrastive lab."""
from __future__ import annotations
import random
from typing import Any
from memory_core import MemoryRecord
from memory_store import tokenize


def memory_utilized(referenced_ids: list[str], retrieved_ids: list[str]) -> bool:
    return bool(set(referenced_ids) & set(retrieved_ids))


def utilization_category(referenced_ids: list[str], retrieved_ids: list[str], outcome_label: str) -> str:
    used = memory_utilized(referenced_ids, retrieved_ids)
    if used and outcome_label == "improved":
        return "beneficial_utilization"
    if used:
        return "used_but_not_beneficial"
    if retrieved_ids:
        return "retrieved_but_ignored"
    return "no_memory_available"


def _choice_score(question: str, choice: str, context: str) -> float:
    signal = set(tokenize(question)) | set(tokenize(context))
    ct = set(tokenize(choice))
    return len(ct & signal) / max(1, len(ct)) if ct else 0.0


def reader_correct(question: str, choices: list[str], gold_idx: int, context: str) -> float:
    if not choices:
        return 0.0
    scores = [_choice_score(question, c, context) for c in choices]
    best = max(range(len(scores)), key=lambda i: scores[i])
    if sum(1 for s in scores if abs(s - scores[best]) < 1e-9) > 1:
        return 0.0
    return 1.0 if best == gold_idx else 0.0


def build_utilization_fixture(locomo_path, longmemeval_dir, longmemeval_files, per_dataset=20, seed=0):
    rng = random.Random(seed)
    items = []
    try:
        from data_loaders import load_locomo, locomo_memory_records, iter_longmemeval_items
        convs = load_locomo(locomo_path)
        all_ans = [qa.answer for c in convs for qa in c.qa_items if qa.answer and qa.category != "adversarial"]
        for conv in convs:
            cr = locomo_memory_records(conv)
            by_id = {str(r["metadata"].get("dia_id", r["id"])): r for r in cr}
            for qa in conv.qa_items:
                if not qa.answer or qa.category == "adversarial" or not qa.evidence_ids:
                    continue
                ev = [by_id[e] for e in qa.evidence_ids if e in by_id]
                if not ev:
                    continue
                d = rng.sample([a for a in all_ans if a != qa.answer], min(3, max(0, len(all_ans)-1))) if len(all_ans) > 4 else []
                while len(d) < 3:
                    d.append("unknown")
                ch = [qa.answer] + d[:3]
                rng.shuffle(ch)
                items.append({"dataset": "LoCoMo", "question": qa.question, "gold_answer": qa.answer,
                    "choices": ch, "gold_idx": ch.index(qa.answer),
                    "evidence_text": "\n".join(str(e["content"]) for e in ev),
                    "corpus": [{"id": r["id"], "content": r["content"], "metadata": r["metadata"]} for r in cr]})
                if sum(1 for i in items if i["dataset"] == "LoCoMo") >= per_dataset:
                    break
            if sum(1 for i in items if i["dataset"] == "LoCoMo") >= per_dataset:
                break
    except Exception:
        pass
    try:
        from pathlib import Path
        lme = []
        ld = Path(longmemeval_dir)
        for fn in longmemeval_files:
            p = ld / fn
            if p.exists():
                lme.extend(iter_longmemeval_items(p, max_items=per_dataset * 3))
        all_a = [it.answer for it in lme if it.answer]
        for it in lme:
            if not it.answer or not it.relevant_ids:
                continue
            ev = [r for r in it.memory_records if str(r["metadata"].get("record_id", r["id"])) in [str(x) for x in it.relevant_ids]]
            if not ev:
                continue
            d = rng.sample([a for a in all_a if a != it.answer], min(3, max(0, len(all_a)-1))) if len(all_a) > 4 else []
            while len(d) < 3:
                d.append("unknown")
            ch = [it.answer] + d[:3]
            rng.shuffle(ch)
            items.append({"dataset": "LongMemEval", "question": it.question, "gold_answer": it.answer,
                "choices": ch, "gold_idx": ch.index(it.answer),
                "evidence_text": "\n".join(str(e["content"]) for e in ev),
                "corpus": [{"id": r["id"], "content": r["content"], "metadata": r["metadata"]} for r in it.memory_records]})
            if sum(1 for i in items if i["dataset"] == "LongMemEval") >= per_dataset:
                break
    except Exception:
        pass
    return items


def run_utilization_lab(fixture, strategy_names, embedder, llm, top_k=5):
    from providers import build_provider
    results = {"n_items": len(fixture), "by_strategy": {}}
    for sn in strategy_names:
        p0 = pgold = pprov = 0.0
        ben = harm = ceil = unres = 0
        for item in fixture:
            ch = item["choices"]
            gi = item["gold_idx"]
            q = item["question"]
            c0 = reader_correct(q, ch, gi, "")
            cg = reader_correct(q, ch, gi, item["evidence_text"])
            prov = build_provider(sn, embedder, llm)
            prov.ingest([MemoryRecord(id=r["id"], content=r["content"], metadata=r.get("metadata", {})) for r in item["corpus"]])
            ret, _ = prov.retrieve(q, top_k=top_k)
            pc = "\n".join(i.content for i in ret)
            cp = reader_correct(q, ch, gi, pc)
            p0 += c0; pgold += cg; pprov += cp
            if not c0 and cp: ben += 1
            elif c0 and not cp: harm += 1
            elif c0 and cp: ceil += 1
            else: unres += 1
        n = max(1, len(fixture))
        results["by_strategy"][sn] = {"n": len(fixture), "p0_rate": round(p0/n, 4),
            "oracle_rate": round(pgold/n, 4), "provider_rate": round(pprov/n, 4),
            "net_gain": round((pprov-p0)/n, 4), "beneficial": ben, "harmful": harm,
            "ceiling": ceil, "unresolved": unres}
    return results


def format_utilization_lab(lab):
    h = "strategy\tp0\toracle\tprovider\tnet_gain\tbenef\tharm\tceil\tunres"
    rows = [h]
    for n, m in lab.get("by_strategy", {}).items():
        rows.append("\t".join(str(x) for x in [n, m["p0_rate"], m["oracle_rate"], m["provider_rate"],
            m["net_gain"], m["beneficial"], m["harmful"], m["ceiling"], m["unresolved"]]))
    return "\n".join(rows)
