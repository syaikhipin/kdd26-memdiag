def retrieval_precision_recall(retrieved_ids: list[str], relevant_ids: list[str]) -> dict[str, float | int]:
    if not retrieved_ids:
        return {"precision": 0.0, "recall": 0.0, "n_relevant": len(relevant_ids)}
    relevant = set(relevant_ids)
    hits = sum(1 for entry_id in retrieved_ids if entry_id in relevant)
    recall = hits / len(relevant) if relevant else 0.0
    return {
        "precision": hits / len(retrieved_ids),
        "recall": recall,
        "n_relevant": len(relevant),
    }
