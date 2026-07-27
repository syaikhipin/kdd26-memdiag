#!/usr/bin/env python3
"""Generate the per-phase tutorial folders under tutorial/.

Every notebook is INDEPENDENT and Colab-ready (own setup cell that auto-clones + installs).
- Phase 1 = 7 notebooks (6 families + capstone)
- Phase 2 = 3 notebooks (diagnose / utilization lab / benchmark)
- Phase 3 = 3 notebooks (loop / run-your-group / aggregate-and-debug)

REPO_URL is defined ONCE here and baked into every notebook's setup cell.
Run: python scripts/build_phase_notebooks.py
"""
import json
from pathlib import Path

TUT = Path(__file__).resolve().parents[1] / "tutorial"
P1 = TUT / "phase1_memory_architectures"
P2 = TUT / "phase2_public_datasets"
P3 = TUT / "phase3_autoresearch"

# ---- change this in ONE place to retarget every notebook's Colab clone -------------------------
REPO_URL = "https://github.com/syaikhipin/kdd26-memdiag"

# Shared, self-contained path resolver (Colab-aware). Included verbatim in every notebook.
RESOLVER_COLAB = f'''# Self-contained setup - works standalone in Google Colab or locally.
import sys, os, subprocess
from pathlib import Path
REPO_URL = "{REPO_URL}" # change in scripts/build_phase_notebooks.py to retarget everywhere
try:
 import google.colab # noqa
 IN_COLAB = True
except Exception:
 IN_COLAB = False
if IN_COLAB:
 repo = Path("/content/kdd26-memdiag")
 if not repo.exists():
 subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo)], check=False)
 SOURCE = repo / "experiment" / "github_submission" / "source"
 subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy", "matplotlib", "pyyaml"], check=False)
else:
 SOURCE = None
 for cand in [Path.cwd(), *Path.cwd().parents]:
 for sub in ("source", "experiment"):
 if (cand / sub / "run.py").exists():
 SOURCE = cand / sub
 break
 if SOURCE:
 break
 if SOURCE is None:
 raise FileNotFoundError("Run from the repo root (or in Colab it auto-clones).")
sys.path.insert(0, str(SOURCE))
os.environ.setdefault("OPENAI_BASE_URL", "https://api.openai.com/v1")
SOURCE_DIR = SOURCE
PROJECT_ROOT = SOURCE.parent
RESULTS_DIR = PROJECT_ROOT / "results"
print("SOURCE_DIR =", SOURCE, "| IN_COLAB =", IN_COLAB)
'''

# Phase 1 setup adds the sample conversation + providers on top of the shared resolver.
SETUP_PHASE1 = RESOLVER_COLAB + '''
from embedder import TfidfHashEmbedder
from llm_client import OfflineLLMClient
from memory_core import MemoryRecord
from providers import build_provider
emb = TfidfHashEmbedder()
try:
 from llm_client import LLMConfig, make_client
 llm = make_client(LLMConfig(backend='openai-compatible' if os.environ.get('OPENAI_API_KEY') else 'offline',
 base_url=os.environ.get('OPENAI_BASE_URL','https://api.openai.com/v1'),
 model='gpt-4o'))
except Exception:
 llm = OfflineLLMClient()
print('LLM backend:', llm.backend, '| API key:', bool(os.environ.get('OPENAI_API_KEY')))
TURNS = [
 ("Alice", "Hi, I am Alice. I work as a data scientist at a health-tech startup in Berlin."),
 ("Bob", "I am Bob, an ML engineer in Athens. I prefer PyTorch."),
 ("Alice", "We deploy on Kubernetes and track runs with Weights and Biases."),
 ("Bob", "Our training run failed last night with CUDA OOM at batch 256."),
 ("Alice", "We hit that before. Reducing batch to 64 and enabling gradient checkpointing fixed it."),
 ("Bob", "Our best val_loss was 0.423 with lr=3e-4 and weight_decay=0.01."),
 ("Alice", "I live in Prenzlauer Berg. My favorite coffee shop is on Kollwitzplatz."),
 ("Bob", "Let us sync next Tuesday at 10am CET."),
]
records = [MemoryRecord(f"t{i}", f"{w}: {t}", {"session_id": "s1" if i < 4 else "s2"}) for i, (w, t) in enumerate(TURNS)]
print("setup OK | turns =", len(TURNS))
'''


def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}


def write_nb(folder: Path, title: str, cells: list[dict], filename: str = "notebook.ipynb") -> None:
 folder.mkdir(parents=True, exist_ok=True)
 nb = {"cells": [md(f"# {title}")] + cells,
 "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
 "language_info": {"name": "python", "version": "3.10"},
 "colab": {"provenance": []}, "accelerator": "none"},
 "nbformat": 4, "nbformat_minor": 5}
 (folder / filename).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
 print("wrote", folder / filename)


def index_readme(folder: Path, title: str, rows: list[tuple[str, str]], note: str = "") -> None:
 lines = [f"# {title}\n", "Each notebook is **self-contained** — open any one in Google Colab (or locally) and run "
 "top-to-bottom. The setup cell auto-clones the repo + installs deps in Colab.\n",
 "| Notebook | Covers |", "|---|---|"]
 for name, covers in rows:
 lines.append(f"| [{name}]({name}) | {covers} |")
 if note:
 lines += ["", note]
 lines += ["", "Attribution: Phase 1 adapted from *Agent Memory Techniques* by Nir Diamant, Apache-2.0." if "Phase 1" in title else "",
 f"REPO_URL (for Colab clone) is set in `scripts/build_phase_notebooks.py`."]
 (folder / "README.md").write_text("\n".join(x for x in lines if x is not None) + "\n", encoding="utf-8")
 print("wrote", folder / "README.md")


# ---- The 30 techniques, by family -------------------------------------------------------------
FAMILIES = [
 ("Short-term memory (01-05) - keep recent turns", [
 ("01", "Conversation Buffer", "Save the full conversation verbatim", "print('01 buffer keeps all', len(TURNS), 'turns verbatim')"),
 ("02", "Sliding Window", "Keep only the last few messages", "print('02 sliding window keeps last 4:', [w for w,_ in TURNS[-4:]])"),
 ("03", "Summary Memory", "Replace old turns with a short summary",
 "try:\n _s = llm.summarize(chr(10).join(f'{w}: {t}' for w,t in TURNS[:5]))\n print('03 summary (LLM):', _s[:80])\nexcept Exception:\n print('03 summary (offline): Alice & Bob discussed ML work; an OOM was fixed by batch 64.')"),
 ("04", "Summary Buffer", "Summarize old turns, keep recent verbatim", "print('04 summary+buffer: summary of old + keep recent:', [w for w,_ in TURNS[-2:]])"),
 ("05", "Token Buffer", "Trim history to a token budget",
 "b=40; out=[]; used=0\nfor w,t in reversed(TURNS):\n n=len(t.split())\n if used+n>b: break\n out.append(w); used+=n\nprint('05 token budget(40) keeps:', list(reversed(out)))"),
 ]),
 ("Long-term memory (06-11) - persist across sessions", [
 ("06", "Vector Store", "Embed turns; retrieve by similarity",
 "p=build_provider('verbatim',emb,llm); p.ingest(records); it,_=p.retrieve('What is Alices job?',3)\nprint('06 vector top:', (it[0].content[:40] if it else None))"),
 ("07", "Entity Memory", "Track facts about people/things",
 "try:\n _facts = llm.extract_facts(chr(10).join(f'{w}: {t}' for w,t in TURNS[:5]))\n print('07 entities/facts (LLM):', _facts[:3])\nexcept Exception:\n import re\n _ents = {n: ' '.join(t for _,t in TURNS if n in t)[:40] for n in ['Alice','Bob']}\n print('07 entities (offline):', _ents)"),
 ("08", "Knowledge Graph", "Triples of how entities connect",
 "triples=[('Alice','works_in','Berlin'),('Bob','role','ML engineer'),('run','failed_with','OOM')]\nprint('08 KG triples:', triples)"),
 ("09", "Episodic Memory", "Store whole interactions + when/where", "p=build_provider('episodic',emb,llm); p.ingest(records); print('09 episodic episodes:', p.size())"),
 ("10", "Semantic Memory", "Extract general facts, dedup/resolve", "p=build_provider('extracted_facts',emb,llm); p.ingest(records); print('10 semantic facts:', p.size())"),
 ("11", "Procedural Memory", "Capture how-to knowledge", "procs=['if OOM: batch 64 + gradient checkpointing','track runs with W&B']\nprint('11 procedural how-to:', procs[0])"),
 ]),
 ("Cognitive architectures (12-19) - human-inspired patterns", [
 ("12", "Working Memory", "Pin/evict context slots by priority", "pinned=[t for w,t in TURNS if 'OOM' in t][:2]\nprint('12 working memory pins:', [p[:30] for p in pinned])"),
 ("13", "Hierarchical Layers", "Hot/warm/cold tiers; promote/demote", "p=build_provider('hybrid',emb,llm); p.ingest(records); print('13 hierarchical tiers size:', p.size())"),
 ("14", "Consolidation", "Merge/dedup/strengthen memories", "seen={}\nfor w,t in TURNS: seen[t[:12]]=seen.get(t[:12],0)+1\nprint('14 consolidation: merged to', len(seen), 'unique prefixes')"),
 ("15", "Compaction", "Compress via summary/entity/distill", "compact=' '.join(t.split()[0] for _,t in TURNS)\nprint('15 compaction:', compact[:60])"),
 ("16", "Self-Reflection", "Agent writes notes on its own actions", "notes=['batch 256 -> OOM (avoid)','lr 3e-4 worked']\nprint('16 self-reflection notes:', notes)"),
 ("17", "Routing", "Pick the memory store by content/intent", "def route(q): return 'temporal' if 'when' in q.lower() else 'semantic'\nprint('17 routing -> ', route('When do they sync?'))"),
 ("18", "Temporal", "Timestamp + recency-weighted retrieval", "import time; now=time.time()\ntagged=[(w, round(now-i*1000,0)) for i,(w,t) in enumerate(TURNS)]\nprint('18 temporal newest:', tagged[-1][0], 'ts', tagged[-1][1])"),
 ("19", "Forgetting & Decay", "Prune by decay/access/relevance", "import time; now=time.time()\nmem=[(w, now-i*5000) for i,(w,t) in enumerate(TURNS)]\nkeep=[m for m in mem if now-m[1]<6000]\nprint('19 forgetting:', len(mem),'->',len(keep),'after 6000s half-life')"),
 ]),
 ("Retrieval & multi-agent (20-23) - find and share memories", [
 ("20", "Retrieval Patterns", "Compare semantic/recency/hybrid scoring",
 "vecs=[emb.embed(t) for _,t in TURNS]; qv=emb.embed('Where does Alice live?')\nsem=max(range(len(TURNS)), key=lambda i: float(vecs[i]@qv))\nprint('20 semantic top:', TURNS[sem][0], '| recency top:', TURNS[-1][0])"),
 ("21", "Cross-Session", "Save/reload agent state across sessions", "import json\nsnap=json.dumps({'last_topic':'OOM fix'})\nprint('21 cross-session snapshot:', snap)"),
 ("22", "Multi-Agent Shared", "Shared stores + message passing", "shared={}\nshared['alice']=['data scientist']; shared['bob']=shared.get('bob',[])+['OOM fix']\nprint('22 shared store:', shared)"),
 ("23", "Memory with Tools", "save/search/forget as callable tools", "store=[]\ndef mem_save(x): store.append(x); return 'saved'\ndef mem_search(q): return store[-1] if store else None\nprint('23 tools:', mem_save('OOM fix'), mem_search('x'))"),
 ]),
 ("Frameworks (24-27) - production memory libraries", [
 ("24", "Graphiti (Zep)", "Time-aware knowledge graphs",
 "print('24 Graphiti: extracts episodes + facts from chat into a TEMPORAL graph.')\nprint(' Use when you need time-aware multi-hop reasoning over conversation history.')\nprint(' Open amt_curated/ for the full notebook (needs a Zep account).')"),
 ("25", "Mem0", "Managed memory layer",
 "print('25 Mem0: managed memory that extracts, stores, and fetches user-specific facts.')\nprint(' Use for personalized AI (remembers user preferences across sessions).')\nprint(' Open upstream AMT notebook 25 (needs a Mem0 account).')"),
 ("26", "Letta / MemGPT", "Self-editing inner/outer memory",
 "print('26 Letta/MemGPT: self-editing memory with inner monologue + memory pressure.')\nprint(' Use when the agent must autonomously manage its own memory budget.')\nprint(' Open upstream AMT notebook 26 (needs a Letta account).')"),
 ("27", "Zep", "Classification + temporal graphs",
 "print('27 Zep: dialog classification + entity extraction + temporal knowledge graphs.')\nprint(' Use for production agents needing structured conversation memory at scale.')\nprint(' Open upstream AMT notebook 27 (needs a Zep account).')"),
 ]),
 ("Evaluation & production (28-30) - measure and deploy", [
 ("28", "Memory Evaluation", "Precision/recall/staleness/contradictions", "def prec(ret,gold):\n g=set(gold); return sum(1 for r in ret if r in g)/max(1,len(ret))\nprint('28 eval precision@2:', prec(['t0','t4'],['t0','t4']))"),
 ("29", "Benchmarks (LoCoMo)", "Run vs LoCoMo/LongMemEval", "print('29 benchmarks (LoCoMo / LongMemEval / MemoryArena): see Phase 2')"),
 ("30", "Production Patterns", "Caching, TTL, sharding, observability", "print('30 production: caching / TTL / sharding / observability - patterns in the README')"),
 ]),
]
FAMILY_META = [
 ("01_short_term_memory.ipynb", "Phase 1A - Short-term memory (01-05)"),
 ("02_long_term_memory.ipynb", "Phase 1B - Long-term memory (06-11)"),
 ("03_cognitive_architectures.ipynb", "Phase 1C - Cognitive architectures (12-19)"),
 ("04_retrieval_multi_agent.ipynb", "Phase 1D - Retrieval & multi-agent (20-23)"),
 ("05_frameworks.ipynb", "Phase 1E - Frameworks (24-27)"),
 ("06_evaluation_production.ipynb", "Phase 1F - Evaluation & production (28-30)"),
]


def build_phase1():
 ATTR = ("Attribution: adapted from *Agent Memory Techniques* by Nir Diamant "
 "(https://github.com/NirDiamant/Agent_Memory_Techniques), Apache-2.0. Inline demos are original.")
 for (fname, title), (label, techniques) in zip(FAMILY_META, FAMILIES):
 cells = [
 md(f"**{title}** - an independent notebook (runnable standalone in Colab or locally).\n\nCovers: {label}.\n\n{ATTR}"),
 md("## 0. Setup (self-contained - run this first)"),
 code(SETUP_PHASE1),
 md(f"## {label}"),
 ]
 for num, name, desc, demo in techniques:
 cells.append(md(f"**{num} - {name}** · {desc}"))
 cells.append(code(f"# Technique {num} - {name}: {desc}\n" + demo))
 write_nb(P1, title, cells, filename=fname)
 capstone = [
 md(f"**Phase 1G - Capstone: the four production architectures** - independent notebook.\n\n{ATTR}"),
 md("## 0. Setup (self-contained)"), code(SETUP_PHASE1),
 md("## Compare all four architectures on one conversation"),
 code("for name in ['no_memory','verbatim','extracted_facts','episodic','hybrid']:\n"
 " p = build_provider(name, emb, llm); p.ingest(records); it,_ = p.retrieve('What is Alices job?', 3)\n"
 " print(f'{name:16s} size={p.size():3d} top=\"{(it[0].content[:38] if it else \"\")}\"')"),
 md("### Bridge to Phase 2\nYou've seen how each technique remembers. Phase 2 asks: on real benchmarks, which failures appear - and how do we measure them honestly?"),
 ]
 write_nb(P1, "Phase 1G - Capstone (4 architectures)", capstone, filename="07_capstone.ipynb")
 index_readme(P1, "Phase 1 - 30 agent-memory techniques (7 independent notebooks)",
 [("01_short_term_memory.ipynb", "01-05 buffer / sliding window / summary / summary-buffer / token"),
 ("02_long_term_memory.ipynb", "06-11 vector / entity / KG / episodic / semantic / procedural"),
 ("03_cognitive_architectures.ipynb", "12-19 working / hierarchical / consolidation / compaction / self-reflection / routing / temporal / forgetting"),
 ("04_retrieval_multi_agent.ipynb", "20-23 retrieval patterns / cross-session / multi-agent / memory-as-tools"),
 ("05_frameworks.ipynb", "24-27 Graphiti / Mem0 / Letta / Zep"),
 ("06_evaluation_production.ipynb", "28-30 evaluation / benchmarks / production"),
 ("07_capstone.ipynb", "the 4 production architectures compared")],
 "Full AMT versions of curated techniques are in `amt_curated/`. **60-min plan:** 0:00-0:08 intro · 0:08-0:52 work through the 6 family notebooks (~8 min each) · 0:52-0:58 capstone · 0:58-1:00 bridge.")


def build_phase2():
 ATTR = "Independent notebook - runs standalone in Colab or locally (offline, no key)."
 nb1 = [
 md(f"**Phase 2A - Diagnostic framework & Exercise 1 (diagnose).** {ATTR}"),
 md("## The 3-probe framework\n- **Probe 1 retrieval:** precision/recall/F1 of retrieved evidence vs gold.\n"
 "- **Probe 2 utilization:** 3-arm contrastive reader (no-context / oracle / provider) - `util != hit`.\n"
 "- **Probe 3 failure root-cause:** retrieval_miss / partial / retrieved-but-unused."),
 md("## 0. Setup (self-contained)"), code(RESOLVER_COLAB),
 md("## Exercise 1 - memory failure diagnosis"),
 code("import subprocess\n"
 "cmd = [sys.executable,'-m','diagnostic_framework','diagnose',\n"
 " '--strategies','verbatim,extracted_facts,episodic,hybrid',\n"
 " '--probes','relevance,utilization,failure','--top_k','5','--max-questions','20']\n"
 "r = subprocess.run(cmd, cwd=str(SOURCE_DIR), capture_output=True, text=True)\n"
 "print(r.stdout[-2200:])\n"
 "if r.returncode: print('STDERR:', r.stderr[-800:])"),
 md("**Read it:** is `util` ever different from `hit`? That gap is the retrieval-vs-utilization failure."),
 ]
 nb2 = [
 md(f"**Phase 2B - The 3-arm utilization lab (independent of retrieval hit).** {ATTR}"),
 md("## 0. Setup (self-contained)"), code(RESOLVER_COLAB),
 md("`provider_rate - p0` is the honest memory gain; `oracle_rate` is the ceiling."),
 code("import pathlib\n"
 "cands = sorted(pathlib.Path(RESULTS_DIR).glob('run_*_real_utilization_lab.tsv')) \\\n"
 " + sorted((RESULTS_DIR/'summaries').glob('*utilization_lab.tsv'))\n"
 "lab = cands[-1] if cands else None\n"
 "print(lab.name, ':\\n'); print(lab.read_text()) if lab else print('Run --mode real first')"),
 md("## Probe one failure in the raw trace"),
 code("import json\n"
 "raws = sorted(pathlib.Path(RESULTS_DIR).glob('run_*_locomo_raw.json'))\n"
 "raw = json.loads(raws[-1].read_text()) if raws else None\n"
 "if raw:\n"
 " rec = next((r for r in raw['records'] if r.get('failure_category')=='retrieval_miss'), raw['records'][0])\n"
 " print('Q:', rec['question'][:70]); print('gold:', rec['evidence_ids'][:5]); print('retrieved:', rec['retrieved_memory_ids'][:5])\n"
 "else: print('(no raw trace - run the diagnose notebook first)')"),
 ]
 nb3 = [
 md(f"**Phase 2C - Exercise 2: benchmarking memory systems.** {ATTR}"),
 md("## 0. Setup (self-contained)"), code(RESOLVER_COLAB),
 md("## Exercise 2 - benchmark providers on MemoryArena"),
 code("import subprocess\n"
 "cmd = [sys.executable,'-m','benchmark_cli','compare',\n"
 " '--providers','verbatim,extracted_facts,episodic,hybrid','--sample_size','20','--benchmark','memoryarena']\n"
 "r = subprocess.run(cmd, cwd=str(SOURCE_DIR), capture_output=True, text=True)\n"
 "print(r.stdout[-2200:])"),
 md("### Discussion\n- Does any architecture dominate across **all** datasets? (Expect: no.)\n- These dataset-dependent tradeoffs motivate Phase 3."),
 ]
 for fname, title, cells in [
 ("01_diagnostic_framework.ipynb", "Phase 2A - Diagnostic framework (Exercise 1)", nb1),
 ("02_utilization_lab.ipynb", "Phase 2B - Utilization lab", nb2),
 ("03_benchmarking.ipynb", "Phase 2C - Benchmarking (Exercise 2)", nb3),
 ]:
 write_nb(P2, title, cells, filename=fname)
 index_readme(P2, "Phase 2 - Public datasets (3 independent notebooks)",
 [("01_diagnostic_framework.ipynb", "3-probe recap + Exercise 1 (diagnose failures)"),
 ("02_utilization_lab.ipynb", "3-arm utilization lab + inspect a failure"),
 ("03_benchmarking.ipynb", "Exercise 2 (benchmark providers) + discussion")],
 "**60-min plan:** 0:00-0:10 lecture · 0:10-0:30 notebook 01 · 0:30-0:40 notebook 02 · 0:40-0:55 notebook 03 · 0:55-1:00 bridge.")


def build_phase3():
 ATTR = "Independent notebook - runs standalone in Colab or locally."
 nb1 = [
 md(f"**Phase 3A - The autoresearch loop & why we split into groups.** {ATTR}"),
 md("Loop: agent edits `train.py` -> 5-min train -> records `val_bpb` -> keep/revise/discard -> repeat. "
 "Memory-debug: remember which configs OOM/regressed so the agent avoids them. A single full sweep hits "
 "CUDA OOM at depth>=12, so we split into groups (deeper models use a smaller batch):\n\n"
 "| Group | configs (depth, batch) |\n|---|---|\n| A | (4,32),(6,32) |\n| B | (8,32),(10,32) |\n| C | (12,16),(14,16) |\n| D | (16,16),(18,16) |"),
 md("## 0. Setup (self-contained)"), code(RESOLVER_COLAB),
 ]
 nb2 = [
 md(f"**Phase 3B - Run YOUR group's slice on Modal L4 (~14.5 min).** {ATTR}"),
 md("## 0. Setup (self-contained)"), code(RESOLVER_COLAB),
 md("Set GROUP to your assigned letter, then run:"),
 code("GROUP = 'A' # A | B | C | D\n"
 "import subprocess\n"
 "script = str(PROJECT_ROOT / 'scripts' / 'build_autoresearch_trace.py')\n"
 "subprocess.run(['modal','run', script, '--group', GROUP], cwd=str(PROJECT_ROOT))\n"
 "print('group', GROUP, 'done -> source/data/topics/autoresearch/leaderboard/'+GROUP+'.json')"),
 ]
 nb3 = [
 md(f"**Phase 3C - Aggregate the sweep + memory-debug loop.** {ATTR}"),
 md("## 0. Setup (self-contained)"), code(RESOLVER_COLAB),
 md("## Aggregate the shared leaderboard"),
 code("import json, pathlib\n"
 "lb = SOURCE_DIR / 'data' / 'topics' / 'autoresearch' / 'leaderboard'\n"
 "rows = []\n"
 "for f in sorted(lb.glob('*.json')):\n"
 " d = json.loads(f.read_text())\n"
 " for t in d.get('trials', []):\n"
 " if t.get('val_bpb') is not None:\n"
 " rows.append((t['config']['depth'], t['config']['device_batch_size'], round(t['val_bpb'],4), d.get('group','?')))\n"
 "rows.sort()\n"
 "for depth,batch,bpb,g in rows: print(f'depth {depth} batch {batch} val_bpb {bpb} (group {g})')\n"
 "if rows: print('Best:', min(rows, key=lambda r: r[2]))\n"
 "else: print('(no group files yet - run the group notebook first, or restore sample leaderboard/A.json)')"),
 md("## Memory-debug loop (keep / revise / discard) + plot"),
 code("if rows:\n"
 " bb=min(r[2] for r in rows)\n"
 " for d,b,p,g in rows: print(f'depth {d} batch {b} val_bpb {p:.3f} ->', 'keep' if p<=bb*1.05 else ('revise' if p<=bb*1.25 else 'discard'))\n"
 "try:\n"
 " import matplotlib.pyplot as plt\n"
 " if rows:\n"
 " rs=sorted(rows); plt.figure(figsize=(7,4)); plt.plot([r[0] for r in rs],[r[2] for r in rs],'o-')\n"
 " plt.xlabel('depth'); plt.ylabel('val_bpb'); plt.title('Phase 3 sweep'); plt.grid(alpha=0.3); plt.show()\n"
 "except Exception as e: print('plot skipped:', e)"),
 md("### Group presentations (3 min each) & wrap-up"),
 ]
 for fname, title, cells in [
 ("01_autoresearch_loop.ipynb", "Phase 3A - Autoresearch loop & group split", nb1),
 ("02_run_your_group.ipynb", "Phase 3B - Run your group's slice (Modal L4)", nb2),
 ("03_aggregate_and_debug.ipynb", "Phase 3C - Aggregate & memory-debug", nb3),
 ]:
 write_nb(P3, title, cells, filename=fname)
 index_readme(P3, "Phase 3 - Autoresearch (3 independent notebooks)",
 [("01_autoresearch_loop.ipynb", "the loop + why we split into groups"),
 ("02_run_your_group.ipynb", "run YOUR group's slice on Modal L4 (~14.5 min)"),
 ("03_aggregate_and_debug.ipynb", "aggregate the sweep + memory-debug loop + plot")],
 "**60-min plan:** 0:00-0:10 notebook 01 · 0:10-0:15 group assignment · 0:15-0:40 notebook 02 (groups) · 0:40-0:48 notebook 03 · 0:48-1:00 presentations.")


def build_tutorial_readme():
 lines = ["# KDD'26 Tutorial - three phases (independent, Colab-ready notebooks)\n",
 "- [Phase 1 - 30 Agent Memory Techniques](phase1_memory_architectures/) - 7 notebooks",
 "- [Phase 2 - Public datasets](phase2_public_datasets/) - 3 notebooks",
 "- [Phase 3 - Autoresearch](phase3_autoresearch/) - 3 notebooks",
 "",
 "Every notebook is self-contained. Open any one in Google Colab and run - the setup cell auto-clones the repo + installs deps. "
 "Phase 1 hands-on adapted from *Agent Memory Techniques* by Nir Diamant, Apache-2.0.",
 "",
 "**Quick start:** [`../RUN.html`](../RUN.html) · [`../SCHEDULE.md`](../SCHEDULE.md)"]
 (TUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
 print("wrote", TUT / "README.md")


if __name__ == "__main__":
 for p in (P1, P2, P3):
 p.mkdir(parents=True, exist_ok=True)
 build_phase1(); build_phase2(); build_phase3(); build_tutorial_readme()
 print("done")
