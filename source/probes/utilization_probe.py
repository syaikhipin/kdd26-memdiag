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
