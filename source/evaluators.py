import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from memory_store import tokenize


SECRET_RE = re.compile(r"(rh" + r"-[A-Za-z0-9_-]+|sk" + r"-[A-Za-z0-9_-]+)")


@dataclass
class EvaluationResult:
    evaluator: str
    semantic_score: float = 0.0
    faithfulness_score: float = 0.0
    context_relevance_score: float = 0.0
    answer_correctness_score: float = 0.0
    passed: bool = False
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    error_redacted: str | None = None

    def to_record_fields(self) -> dict[str, Any]:
        data = asdict(self)
        return {
            "semantic_evaluator": data.pop("evaluator"),
            "semantic_score": data.pop("semantic_score"),
            "faithfulness_score": data.pop("faithfulness_score"),
            "context_relevance_score": data.pop("context_relevance_score"),
            "answer_correctness_score": data.pop("answer_correctness_score"),
            "semantic_pass": data.pop("passed"),
            "semantic_reason": data.pop("reason"),
            "semantic_details": data.pop("details"),
            "semantic_error_redacted": data.pop("error_redacted"),
        }


class BaseEvaluator:
    name = "base"

    def evaluate(self, record: dict[str, Any]) -> EvaluationResult:
        raise NotImplementedError


def redact_secret(text: Any) -> str:
    return SECRET_RE.sub(lambda match: match.group(0).split("-", 1)[0] + "-REDACTED", str(text))


class OfflineSemanticEvaluator(BaseEvaluator):
    name = "offline_semantic"

    def evaluate(self, record: dict[str, Any]) -> EvaluationResult:
        gold = str(record.get("gold_answer") or record.get("answer") or "")
        answer = str(record.get("answer") or "")
        question = str(record.get("question") or record.get("title") or "")
        contexts = [str(item) for item in record.get("retrieved_texts", [])]
        context_text = "\n".join(contexts)

        answer_correctness = _overlap(gold, answer)
        if answer in {"retrieved_evidence_answerable", "insufficient_retrieved_evidence"}:
            answer_correctness = float(bool(record.get("evidence_hit") or record.get("memory_utilized")))
        context_relevance = max(_overlap(question, context_text), float(record.get("retrieval_recall", 0.0)))
        if answer in {"retrieved_evidence_answerable", "insufficient_retrieved_evidence"}:
            faithfulness = float(bool(record.get("evidence_hit") and record.get("memory_utilized")))
        else:
            faithfulness = _overlap(answer, context_text) if answer else float(bool(record.get("memory_utilized")))
        semantic = round((answer_correctness + context_relevance + faithfulness) / 3.0, 4)
        passed = semantic >= 0.5 or bool(record.get("evidence_hit"))
        return EvaluationResult(
            evaluator=self.name,
            semantic_score=semantic,
            faithfulness_score=round(faithfulness, 4),
            context_relevance_score=round(context_relevance, 4),
            answer_correctness_score=round(answer_correctness, 4),
            passed=passed,
            reason="offline token-overlap semantic proxy",
            details={
                "gold_tokens": len(set(tokenize(gold))),
                "answer_tokens": len(set(tokenize(answer))),
                "context_tokens": len(set(tokenize(context_text))),
            },
        )


class RhesisEvaluator(BaseEvaluator):
    name = "rhesis"

    def __init__(self, api_key_env: str = "RHESIS_API_KEY", model: str | None = None, base_url: str | None = None):
        self.api_key_env = api_key_env
        self.model = model or "rhesis"
        self.base_url = base_url or os.environ.get("RHESIS_BASE_URL")

    def evaluate(self, record: dict[str, Any]) -> EvaluationResult:
        if not os.environ.get(self.api_key_env):
            return EvaluationResult(
                evaluator=self.name,
                reason=f"skipped: {self.api_key_env} is not set",
                error_redacted=f"missing {self.api_key_env}",
            )
        try:
            from rhesis.sdk.models import get_model
        except Exception as exc:
            return EvaluationResult(
                evaluator=self.name,
                reason="skipped: rhesis-sdk is not installed or importable",
                error_redacted=redact_secret(exc),
            )
        try:
            if self.base_url:
                os.environ["RHESIS_BASE_URL"] = self.base_url
            model = get_model(self.model)
            prompt = _judge_prompt(record)
            raw = model.generate(prompt=prompt)
            parsed = _parse_numeric_judge(raw)
            return EvaluationResult(
                evaluator=self.name,
                semantic_score=parsed["score"],
                faithfulness_score=parsed["score"],
                context_relevance_score=parsed["score"],
                answer_correctness_score=parsed["score"],
                passed=parsed["score"] >= 0.5,
                reason=parsed["rationale"],
                details={"rhesis_model": self.model, "rhesis_base_url_present": bool(self.base_url)},
            )
        except Exception as exc:
            return EvaluationResult(
                evaluator=self.name,
                reason="rhesis evaluation failed",
                error_redacted=redact_secret(exc),
            )


class SemanticaEvaluator(BaseEvaluator):
    name = "semantica"

    def __init__(self, mode: str = "extract"):
        self.mode = mode

    def evaluate(self, record: dict[str, Any]) -> EvaluationResult:
        try:
            from semantica.semantic_extract import NERExtractor, TripletExtractor
        except Exception as exc:
            return EvaluationResult(
                evaluator=self.name,
                reason="skipped: semantica is not installed or semantic_extract API is unavailable",
                error_redacted=redact_secret(exc),
            )
        try:
            gold = str(record.get("gold_answer") or "")
            context = "\n".join(str(item) for item in record.get("retrieved_texts", []))
            text = f"Question: {record.get('question', '')}\nGold: {gold}\nContext: {context}"
            entities = _safe_extract_entities(NERExtractor(), text)
            triplets = _safe_extract_triplets(TripletExtractor(), text)
            gold_entities = _entity_labels(_safe_extract_entities(NERExtractor(), gold))
            context_entities = _entity_labels(entities)
            entity_coverage = _set_overlap(gold_entities, context_entities)
            triplet_signal = min(1.0, len(triplets) / 10.0)
            retrieval_signal = float(record.get("retrieval_recall", 0.0))
            semantic = round((entity_coverage + triplet_signal + retrieval_signal) / 3.0, 4)
            return EvaluationResult(
                evaluator=self.name,
                semantic_score=semantic,
                faithfulness_score=round(entity_coverage, 4),
                context_relevance_score=round(max(entity_coverage, retrieval_signal), 4),
                answer_correctness_score=round(entity_coverage, 4),
                passed=semantic >= 0.5,
                reason="semantica entity/triplet provenance proxy",
                details={
                    "semantica_mode": self.mode,
                    "entities": list(context_entities)[:25],
                    "triplet_count": len(triplets),
                    "gold_entity_count": len(gold_entities),
                    "context_entity_count": len(context_entities),
                },
            )
        except Exception as exc:
            return EvaluationResult(
                evaluator=self.name,
                reason="semantica evaluation failed",
                error_redacted=redact_secret(exc),
            )


def make_evaluators(eval_backend: str, *, rhesis_api_key_env: str = "RHESIS_API_KEY", rhesis_model: str | None = None, rhesis_base_url: str | None = None, semantica_mode: str = "extract") -> list[BaseEvaluator]:
    if eval_backend == "offline":
        return [OfflineSemanticEvaluator()]
    if eval_backend == "rhesis":
        return [RhesisEvaluator(rhesis_api_key_env, rhesis_model, rhesis_base_url)]
    if eval_backend == "semantica":
        return [SemanticaEvaluator(semantica_mode)]
    if eval_backend == "all":
        return [
            OfflineSemanticEvaluator(),
            RhesisEvaluator(rhesis_api_key_env, rhesis_model, rhesis_base_url),
            SemanticaEvaluator(semantica_mode),
        ]
    return []


def _overlap(a: str, b: str) -> float:
    left = set(tokenize(a))
    right = set(tokenize(b))
    if not left or not right:
        return 0.0
    return round(len(left & right) / len(left), 4)


def _set_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return round(len(left & right) / len(left), 4)


def _judge_prompt(record: dict[str, Any]) -> str:
    return "\n".join([
        "Evaluate this autonomous-agent memory answer on a 0.0 to 1.0 scale.",
        "Return JSON with fields score and rationale.",
        f"Question: {record.get('question', '')}",
        f"Gold answer: {record.get('gold_answer', '')}",
        f"Candidate answer: {record.get('answer', '')}",
        "Retrieved context:",
        "\n".join(str(item) for item in record.get("retrieved_texts", [])[:5]),
    ])


def _parse_numeric_judge(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        payload = raw
    else:
        text = str(raw)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"(0(?:\.\d+)?|1(?:\.0+)?)", text)
            payload = {"score": float(match.group(1)) if match else 0.0, "rationale": text[:500]}
    score = float(payload.get("score", 0.0))
    if score > 1.0:
        score = score / 5.0 if score <= 5.0 else score / 100.0
    return {"score": round(max(0.0, min(1.0, score)), 4), "rationale": str(payload.get("rationale", ""))[:500]}


def _safe_extract_entities(extractor: Any, text: str) -> list[Any]:
    try:
        return list(extractor.extract_entities(text))
    except Exception:
        return []


def _safe_extract_triplets(extractor: Any, text: str) -> list[Any]:
    try:
        return list(extractor.extract_triplets(text))
    except Exception:
        return []


def _entity_labels(entities: list[Any]) -> set[str]:
    labels = set()
    for entity in entities:
        if isinstance(entity, dict):
            value = entity.get("label") or entity.get("text") or entity.get("name") or entity.get("id")
        else:
            value = getattr(entity, "label", None) or getattr(entity, "text", None) or getattr(entity, "name", None) or str(entity)
        if value:
            labels.update(tokenize(str(value)))
    return labels
