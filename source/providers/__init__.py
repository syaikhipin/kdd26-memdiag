"""Memory-provider registry."""
from __future__ import annotations
from typing import Any
from memory_core import MemoryProvider
from .verbatim import VerbatimProvider
from .extracted_facts import ExtractedFactsProvider
from .episodic import EpisodicProvider
from .hybrid import HierarchicalProvider
from .no_memory import NoMemoryProvider

REGISTRY = {
    "no_memory": NoMemoryProvider,
    "verbatim": VerbatimProvider,
    "extracted_facts": ExtractedFactsProvider,
    "episodic": EpisodicProvider,
    "hybrid": HierarchicalProvider,
}
ALL_PROVIDERS = list(REGISTRY)


def build_provider(name: str, embedder, llm, **opts: Any) -> MemoryProvider:
    if name not in REGISTRY:
        raise ValueError(f"Unknown memory provider '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](embedder, llm, **opts)
