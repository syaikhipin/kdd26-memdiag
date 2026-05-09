import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class Turn:
    dia_id: str
    speaker: str
    text: str


@dataclass(frozen=True)
class Session:
    session_id: str
    timestamp: str | None
    turns: list[Turn]


@dataclass(frozen=True)
class QAItem:
    question: str
    answer: str
    category: str
    evidence_ids: list[str]


@dataclass(frozen=True)
class Conversation:
    conv_id: str
    sessions: list[Session]
    qa_items: list[QAItem]


@dataclass(frozen=True)
class MemoryBenchmarkItem:
    dataset: str
    item_id: str
    question: str
    answer: str
    category: str
    memory_records: list[dict[str, Any]]
    relevant_ids: list[str]


REAL_DATA_DIR = Path(__file__).resolve().parent / "data" / "real"
DEFAULT_LONGMEMEVAL_DIR = REAL_DATA_DIR / "longmemeval"
DEFAULT_MEMORYARENA_DIR = REAL_DATA_DIR / "memoryarena"
DEFAULT_LCBENCH_DIR = REAL_DATA_DIR / "lcbench"
DEFAULT_HPOBENCH_DIR = REAL_DATA_DIR / "hpobench"


CATEGORY_MAP = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal_reasoning",
    4: "open_domain",
    5: "adversarial",
}


def dataset_registry(locomo_path: Path) -> list[dict[str, Any]]:
    longmemeval_files = sorted(DEFAULT_LONGMEMEVAL_DIR.glob("*.json")) if DEFAULT_LONGMEMEVAL_DIR.exists() else []
    memoryarena_files = sorted(DEFAULT_MEMORYARENA_DIR.glob("*.jsonl")) if DEFAULT_MEMORYARENA_DIR.exists() else []
    lcbench_available = any(DEFAULT_LCBENCH_DIR.iterdir()) if DEFAULT_LCBENCH_DIR.exists() else False
    hpobench_available = any(DEFAULT_HPOBENCH_DIR.iterdir()) if DEFAULT_HPOBENCH_DIR.exists() else False
    return [
        {
            "name": "LCBench",
            "source": "automl/LCBench",
            "tutorial_use": "Autonomous research traces and HPO experiment memories",
            "available_local": lcbench_available,
            "default_path": str(DEFAULT_LCBENCH_DIR) if lcbench_available else None,
            "status": "directory present but no downloaded data" if not lcbench_available else "available locally",
        },
        {
            "name": "LoCoMo",
            "source": "snap-research/locomo",
            "tutorial_use": "Conversational long-horizon memory evaluation",
            "available_local": locomo_path.exists(),
            "default_path": str(locomo_path),
            "status": "available locally" if locomo_path.exists() else "listed in PDF; local file missing",
        },
        {
            "name": "MemoryArena",
            "source": "ZexueHe/memoryarena",
            "tutorial_use": "Multi-session agent memory tasks",
            "available_local": bool(memoryarena_files),
            "default_path": str(DEFAULT_MEMORYARENA_DIR) if memoryarena_files else None,
            "files": [path.name for path in memoryarena_files],
            "status": "available locally" if memoryarena_files else "listed in PDF; local data not found",
        },
        {
            "name": "HPOBench",
            "source": "automl/HPOBench",
            "tutorial_use": "HPO experiment traces",
            "available_local": hpobench_available,
            "default_path": str(DEFAULT_HPOBENCH_DIR) if hpobench_available else None,
            "status": "directory present but no downloaded data" if not hpobench_available else "available locally",
        },
        {
            "name": "LongMemEval",
            "source": "xiaowu0162/longmemeval-cleaned",
            "tutorial_use": "Long-term memory benchmarking",
            "available_local": bool(longmemeval_files),
            "default_path": str(DEFAULT_LONGMEMEVAL_DIR) if longmemeval_files else None,
            "files": [path.name for path in longmemeval_files],
            "status": "available locally" if longmemeval_files else "listed in PDF; local data not found",
        },
    ]


def load_locomo(path: Path) -> list[Conversation]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    conversations = []
    for conv_idx, conv_data in enumerate(raw):
        sessions = []
        conv = conv_data.get("conversation", {})
        for session_key in sorted((k for k in conv if re.match(r"session_\d+$", k)), key=_session_sort_key):
            turns = []
            for turn in conv.get(session_key, []):
                turns.append(Turn(
                    dia_id=str(turn.get("dia_id", "")),
                    speaker=str(turn.get("speaker", "unknown")),
                    text=str(turn.get("text", "")),
                ))
            sessions.append(Session(
                session_id=session_key,
                timestamp=conv.get(f"{session_key}_date_time"),
                turns=turns,
            ))

        qa_items = []
        for qa in conv_data.get("qa", []):
            category_id = qa.get("category")
            evidence = qa.get("evidence", []) or []
            qa_items.append(QAItem(
                question=str(qa.get("question", "")),
                answer=str(qa.get("answer", "")),
                category=CATEGORY_MAP.get(category_id, str(category_id)),
                evidence_ids=[str(item) for item in evidence],
            ))
        conversations.append(Conversation(
            conv_id=str(conv_data.get("sample_id", conv_idx)),
            sessions=sessions,
            qa_items=qa_items,
        ))
    return conversations


def _session_sort_key(name: str) -> int:
    return int(name.split("_")[-1])


def format_session_as_text(session: Session) -> str:
    lines = []
    if session.timestamp:
        lines.append(f"[Date: {session.timestamp}]")
    lines.extend(f"{turn.speaker}: {turn.text}" for turn in session.turns)
    return "\n".join(lines)


def iter_locomo_questions(conversation: Conversation, max_questions: int | None):
    items = [qa for qa in conversation.qa_items if qa.category != "adversarial"]
    if max_questions is not None:
        items = items[:max_questions]
    for idx, qa in enumerate(items):
        yield idx, qa


def locomo_memory_records(conversation: Conversation) -> list[dict[str, Any]]:
    records = []
    for session in conversation.sessions:
        for turn in session.turns:
            records.append({
                "id": turn.dia_id,
                "content": f"{turn.speaker}: {turn.text}",
                "metadata": {
                    "conv_id": conversation.conv_id,
                    "session_id": session.session_id,
                    "record_id": turn.dia_id,
                    "dia_id": turn.dia_id,
                    "speaker": turn.speaker,
                    "timestamp": session.timestamp,
                    "entry_type": "locomo_turn",
                },
            })
    return records


def iter_locomo_benchmark_items(path: Path, max_conversations: int | None, max_questions: int | None) -> Iterator[MemoryBenchmarkItem]:
    conversations = load_locomo(path)
    if max_conversations is not None:
        conversations = conversations[:max_conversations]
    for conversation in conversations:
        records = locomo_memory_records(conversation)
        for question_idx, qa in iter_locomo_questions(conversation, max_questions):
            yield MemoryBenchmarkItem(
                dataset="LoCoMo",
                item_id=f"{conversation.conv_id}:{question_idx}",
                question=qa.question,
                answer=qa.answer,
                category=qa.category,
                memory_records=records,
                relevant_ids=qa.evidence_ids,
            )


def iter_longmemeval_items(path: Path, max_items: int | None = None) -> Iterator[MemoryBenchmarkItem]:
    for idx, row in enumerate(_iter_json_array(path)):
        if max_items is not None and idx >= max_items:
            break
        item_id = str(row.get("question_id", idx))
        session_ids = [str(value) for value in row.get("haystack_session_ids", [])]
        dates = row.get("haystack_dates", [])
        sessions = row.get("haystack_sessions", [])
        records = []
        for session_idx, session in enumerate(sessions):
            session_id = session_ids[session_idx] if session_idx < len(session_ids) else f"{item_id}:session:{session_idx}"
            date = dates[session_idx] if session_idx < len(dates) else None
            content = _messages_to_text(session)
            records.append({
                "id": session_id,
                "content": content,
                "metadata": {
                    "dataset": "LongMemEval",
                    "question_id": item_id,
                    "record_id": session_id,
                    "session_id": session_id,
                    "timestamp": date,
                    "has_answer": _session_has_answer(session),
                    "entry_type": "longmemeval_session",
                },
            })
        yield MemoryBenchmarkItem(
            dataset="LongMemEval",
            item_id=item_id,
            question=str(row.get("question", "")),
            answer=str(row.get("answer", "")),
            category=str(row.get("question_type", "unknown")),
            memory_records=records,
            relevant_ids=[str(value) for value in row.get("answer_session_ids", [])],
        )


def iter_memoryarena_items(path: Path, max_items: int | None = None) -> Iterator[MemoryBenchmarkItem]:
    emitted = 0
    task_name = path.stem
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            questions = row.get("questions", []) or []
            answers = row.get("answers", []) or []
            base_records = _memoryarena_records(task_name, row, line_idx)
            for question_idx, question in enumerate(questions):
                if max_items is not None and emitted >= max_items:
                    return
                answer = answers[question_idx] if question_idx < len(answers) else ""
                relevant = _memoryarena_relevant_ids(base_records, question_idx)
                yield MemoryBenchmarkItem(
                    dataset="MemoryArena",
                    item_id=f"{task_name}:{row.get('id', line_idx)}:{question_idx}",
                    question=str(question),
                    answer=_stringify(answer),
                    category=str(row.get("category") or row.get("paper_name") or task_name),
                    memory_records=base_records,
                    relevant_ids=relevant,
                )
                emitted += 1


def _iter_json_array(path: Path) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buffer = ""
        in_array = False
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            buffer += chunk
            pos = 0
            if not in_array:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer) and buffer[pos] == "[":
                    pos += 1
                    in_array = True
            while in_array:
                while pos < len(buffer) and buffer[pos] in " \r\n\t,":
                    pos += 1
                if pos < len(buffer) and buffer[pos] == "]":
                    return
                try:
                    item, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    break
                yield item
                pos = end
            buffer = buffer[pos:]


def _messages_to_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return _stringify(messages)
    lines = []
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role", "speaker")
            content = message.get("content", "")
            lines.append(f"{role}: {content}")
        else:
            lines.append(str(message))
    return "\n".join(lines)


def _session_has_answer(messages: Any) -> bool:
    return any(isinstance(message, dict) and bool(message.get("has_answer")) for message in messages if isinstance(messages, list))


def _memoryarena_records(task_name: str, row: dict[str, Any], line_idx: int) -> list[dict[str, Any]]:
    records = []
    row_id = row.get("id", line_idx)
    if isinstance(row.get("backgrounds"), list):
        for idx, background in enumerate(row["backgrounds"]):
            records.append(_record("MemoryArena", f"{task_name}:{row_id}:background:{idx}", background, task_name, "memoryarena_background", row, idx))
    if isinstance(row.get("base_person"), dict):
        base = row["base_person"]
        records.append(_record("MemoryArena", f"{task_name}:{row_id}:base_person", base, task_name, "memoryarena_profile", row, None))
        for idx, plan in enumerate(base.get("daily_plans", []) or []):
            records.append(_record("MemoryArena", f"{task_name}:{row_id}:daily_plan:{idx}", plan, task_name, "memoryarena_daily_plan", row, idx))
    if not records:
        context_parts = []
        for key, value in row.items():
            if key not in {"questions", "answers"}:
                context_parts.append(f"{key}: {_stringify(value)}")
        records.append(_record("MemoryArena", f"{task_name}:{row_id}:context", "\n".join(context_parts), task_name, "memoryarena_context", row, None))
    return records


def _record(dataset: str, record_id: str, content: Any, task_name: str, entry_type: str, row: dict[str, Any], index: int | None) -> dict[str, Any]:
    return {
        "id": record_id,
        "content": _stringify(content),
        "metadata": {
            "dataset": dataset,
            "task_name": task_name,
            "record_id": record_id,
            "source_id": row.get("id"),
            "background_idx": index,
            "entry_type": entry_type,
        },
    }


def _memoryarena_relevant_ids(records: list[dict[str, Any]], question_idx: int) -> list[str]:
    indexed = [record for record in records if record["metadata"].get("background_idx") == question_idx]
    if indexed:
        return [record["id"] for record in indexed]
    return [record["id"] for record in records]


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
