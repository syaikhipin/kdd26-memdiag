from .no_memory import NoMemoryStrategy
from .verbatim import VerbatimStrategy
from .extracted_facts import ExtractedFactsStrategy
from .episodic import EpisodicStrategy
from .hybrid import HybridStrategy


ALL_STRATEGIES = {
    "no_memory": NoMemoryStrategy,
    "verbatim": VerbatimStrategy,
    "extracted_facts": ExtractedFactsStrategy,
    "episodic": EpisodicStrategy,
    "hybrid": HybridStrategy,
}
