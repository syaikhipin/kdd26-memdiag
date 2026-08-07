"""Cognitive-constraint memory strategy ("CML-lite").

A pure-Python, offline re-implementation of the key differentiator of the
Cognitive Memory Layer (https://github.com/avinash-mall/CognitiveMemoryLayer):
constraint *extraction* at write time + constraint-*aware* retrieval that
surfaces stored constraints even when the query has zero lexical overlap.

The full CML system is neuro-inspired and needs Docker + Postgres + Neo4j +
Redis + ~25 GB of model weights, so it cannot run in a Colab tutorial slot.
This strategy captures CML's cognitive core in ~90 lines so it can be
diagnosed with the tutorial's existing 3-probe / LoCoMo harness *offline* --
the same low-resource ethos as the other providers.

Reference patterns reused here:
- BaseMemoryStrategy interface: ``strategies/base.py``
- MemoryStore.add / retrieve: ``memory_store.py``
- CAUSAL rule inference ("depth>=D at batch B -> OOM"):
  ``github_submission/scripts/phase3_diagnostic_v3.py``
"""
from __future__ import annotations

import re

from models import Episode, ResearchTask
from .base import BaseMemoryStrategy

# ---------------------------------------------------------------------------
# Write path: constraint extraction
# ---------------------------------------------------------------------------
# CML defines 5 cognitive constraint types (goal / state / value / causal /
# policy). The real system extracts them with one LLM call per chunk; this
# deterministic keyword classifier mirrors CML's own regex fallback mode.

CONSTRAINT_TYPES = ("POLICY", "CAUSAL", "GOAL", "VALUE", "STATE")

_CONSTRAINT_PATTERNS: dict[str, list[str]] = {
    "POLICY": [
        r"\ballergic\b", r"\bvegetarian\b", r"\bvegan\b", r"\bmust not\b",
        r"\bmustn't\b", r"\bnever\b", r"\bdo not\b", r"\bdon't\b",
        r"\bavoid\b", r"\bforbidden\b", r"\bnot allowed\b",
    ],
    "CAUSAL": [
        r"\bcaused\b", r"\bled to\b", r"\bresulted in\b", r"\btriggered\b",
        r"\bcuda oom\b", r"\boom\b", r"\bcrashed\b", r"\bregressed\b",
        r"\bdegraded\b", r"\bfailed because\b",
    ],
    "GOAL": [
        r"\bi want\b", r"\bi'd like\b", r"\btrying to\b", r"\baim to\b",
        r"\bgoal is\b", r"\bmy objective\b", r"\bhope to\b",
    ],
    "VALUE": [
        r"\bi value\b", r"\bi prefer\b", r"\bi care about\b",
        r"\bi prioritise\b", r"\bi prioritize\b", r"\bmatters most\b",
    ],
    "STATE": [
        r"\bis down\b", r"\bis broken\b", r"\bis offline\b", r"\boutage\b",
        r"\bunavailable\b", r"\bis degraded\b", r"\bserver failed\b",
    ],
}

_COMPILED: dict[str, list[re.Pattern[str]]] = {
    ctype: [re.compile(p, re.IGNORECASE) for p in patterns]
    for ctype, patterns in _CONSTRAINT_PATTERNS.items()
}

# Cues that the query is asking for a choice/recommendation (surfaces GOAL/VALUE)
# or about taking a config/action (surfaces CAUSAL, e.g. the Phase-3 OOM rule).
_RECOMMENDATION_CUES = (
    "recommend", "suggest", "what should", "which", "best", "propose",
    "choose", "pick", "next", "try", "order", "eat",
)
_ACTION_CUES = (
    "run", "deploy", "depth", "batch", "config", "use", "apply", "train",
    "increase", "set", "launch", "scale",
)


def classify_constraint(text: str) -> str | None:
    """Return the first matching constraint type for *text*, or ``None``.

    Exposed at module level so the notebook can demonstrate extraction on raw
    strings (e.g. ``"I'm allergic to shellfish" -> "POLICY"``) without building
    a full ``Episode``. Order matters: POLICY/CAUSAL are checked before STATE
    so a failure with an explicit cause is tagged CAUSAL, not STATE.
    """
    if not text:
        return None
    for ctype in CONSTRAINT_TYPES:
        if any(rx.search(text) for rx in _COMPILED[ctype]):
            return ctype
    return None


class CognitiveConstraintStrategy(BaseMemoryStrategy):
    """Memory that understands *constraints*, not just keywords.

    Write path (``ingest_episode``): classify each stored chunk into a
    constraint type and tag it in metadata. A failed episode with no explicit
    cause is treated as a CAUSAL rule (mirrors Phase-3's OOM inference).

    Read path (``retrieve``): standard lexical retrieval *plus* any stored
    constraint that is relevant to the decision -- surfaced regardless of
    cosine overlap with the query. This is what lets it beat plain RAG on
    zero-overlap cases such as storing "I'm allergic to shellfish" and later
    asking "recommend a restaurant".
    """

    name = "cognitive_constraint"

    # -- harness-compatible write path (used by run_strategy / benchmarks) -----
    def ingest_episode(self, task: ResearchTask, episode: Episode) -> list[str]:
        content = (
            f"Task {task.task_id} ({task.domain}) step {episode.step_idx}: "
            f"action '{episode.proposed_action}'. Rationale: {episode.rationale}. "
            f"Outcome: {episode.outcome_label}, score={episode.outcome_score:.3f}, "
            f"failure={episode.failure_mode or 'none'}."
        )
        ctype = classify_constraint(content)
        if ctype is None and episode.failure_mode:
            # No keyword cue, but the episode failed -> generalise it into a
            # CAUSAL rule (phase3_diagnostic_v3.py does the same for OOM).
            ctype = "CAUSAL"
        return [
            self.store.add(
                content,
                metadata={
                    "action": episode.proposed_action,
                    "outcome_label": episode.outcome_label,
                    "score": episode.outcome_score,
                    "failure_mode": episode.failure_mode,
                    "constraint_type": ctype,
                },
                source_episode=episode.episode_id,
                source_task=task.task_id,
                entry_type="constraint" if ctype else "memory",
            )
        ]

    # -- cognitive read path: constraint-aware reranking ----------------------
    def retrieve(self, query: str, task: ResearchTask, top_k: int):
        # 1. Standard lexical retrieval (cosine token-overlap), as in the base store.
        retrieved, latency_ms = self.store.retrieve(query, top_k=top_k)

        # 2. Cognitive layer: surface constraints relevant to *this decision*
        #    independent of lexical overlap. POLICY/STATE are safety-critical
        #    and always surfaced; CAUSAL on action queries (the OOM rule fires
        #    when the agent considers a new config); GOAL/VALUE on recommendations.
        query_lower = query.lower()
        is_recommendation = any(cue in query_lower for cue in _RECOMMENDATION_CUES)
        is_action = any(cue in query_lower for cue in _ACTION_CUES)

        surfaced: list[dict] = []
        seen: set[str] = set()
        for entry in self.store.entries:
            ctype = entry.metadata.get("constraint_type")
            if not ctype:
                continue
            relevant = (
                ctype in ("POLICY", "STATE")
                or (ctype == "CAUSAL" and is_action)
                or (ctype in ("GOAL", "VALUE") and is_recommendation)
            )
            if relevant and entry.id not in seen:
                surfaced.append({"entry": entry, "score": 1.0})
                seen.add(entry.id)

        # 3. Constraints first (cognitive rerank), then lexical hits, de-duped.
        merged = list(surfaced)
        for item in retrieved:
            if item["entry"].id not in seen:
                merged.append(item)
                seen.add(item["entry"].id)
        return merged, latency_ms

    # -- demo-friendly API mirroring the real CML Python SDK -----------------
    def write(self, content: str, task_id: str = "demo") -> str:
        """Write a raw text chunk, extracting any constraint. Notebook/demo use."""
        ctype = classify_constraint(content)
        return self.store.add(
            content,
            metadata={"constraint_type": ctype},
            source_task=task_id,
            entry_type="constraint" if ctype else "memory",
        )

    def read(self, query: str, top_k: int = 5):
        """SDK-style read alias (task is unused by the cognitive read path)."""
        return self.retrieve(query, task=None, top_k=top_k)
