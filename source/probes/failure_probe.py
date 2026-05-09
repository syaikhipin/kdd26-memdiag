def classify_failure(
    failure_mode: str | None,
    retrieved_ids: list[str],
    relevant_ids: list[str],
    referenced_ids: list[str],
    redundant_action: bool,
) -> str:
    if failure_mode:
        return failure_mode
    if redundant_action:
        return "redundant_experiment"
    if relevant_ids and not retrieved_ids:
        return "retrieval_miss"
    if relevant_ids and not (set(retrieved_ids) & set(relevant_ids)):
        return "misleading_memory"
    if retrieved_ids and not referenced_ids:
        return "retrieved_but_ignored"
    return "none"
