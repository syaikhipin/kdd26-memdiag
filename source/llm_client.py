"""LLM client: OfflineLLMClient (no key) + OpenAICompatibleClient (API)."""
from __future__ import annotations
import json, os, re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from memory_store import tokenize


@dataclass(frozen=True)
class LLMConfig:
    backend: str = "offline"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    timeout_s: int = 60
    max_retries: int = 2
    def resolved_api_key(self):
        return self.api_key or os.environ.get(self.api_key_env)
    def safe_dict(self):
        return {"backend": self.backend, "base_url": self.base_url, "model": self.model,
                "api_key_env": self.api_key_env, "api_key_present": bool(self.resolved_api_key())}


class LLMClient(ABC):
    backend = "abstract"
    @abstractmethod
    def chat_text(self, system, prompt): ...
    @abstractmethod
    def chat_json(self, system, prompt): ...
    @abstractmethod
    def extract_facts(self, text): ...
    @abstractmethod
    def summarize(self, text): ...
    @abstractmethod
    def extract_topic_tags(self, text, n=3): ...
    @abstractmethod
    def judge(self, system, prompt): ...


_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_FP = re.compile(r"\b(I'm|I've|I'd|I'll|I have|I like|I prefer|I|my|me)\b", re.IGNORECASE)  # longest-first: every multi-token contraction/phrase must precede bare 'I' or it shadows them ('I'd' -> 'the user'd')


def _norm(s):
    s = s.strip().strip("\"'.,")
    def _rw(m):
        w = m.group(0).lower()
        return {"i":"the user","i'm":"the user is","i'd":"the user would","i'll":"the user will",
                "i have":"the user has","i've":"the user has","i like":"the user likes","i prefer":"the user prefers",
                "my":"the user's","me":"the user"}.get(w, w)
    s = _FP.sub(_rw, s)
    w = s.split()
    if len(w) > 20: w = w[:20] + ["..."]
    return " ".join(w)


class OfflineLLMClient(LLMClient):
    backend = "offline"
    def chat_text(self, s, p): return ""
    def chat_json(self, s, p): return {}
    def extract_facts(self, text):
        if not text or not text.strip(): return []
        ss = [x for x in _SENT.split(text) if len(x.split()) >= 3]
        fs = [_norm(x) for x in ss if _norm(x)]
        return fs[:6] if fs else ([_norm(text)] if _norm(text) else [])
    def summarize(self, text):
        if not text or not text.strip(): return ""
        ss = [x for x in _SENT.split(text) if x.strip()]
        if len(ss) <= 2: return " ".join(_norm(x) for x in ss)
        from collections import Counter
        freq = Counter(tokenize(text))
        scored = sorted(ss, key=lambda x: sum(freq.get(t,0) for t in tokenize(x))/max(1,len(tokenize(x))), reverse=True)
        # lead sentence + best *different* sentence; without dedup, lead==best emits it twice
        _pick = []
        for x in list(ss[:1]) + list(scored):
            if x not in _pick: _pick.append(x)
            if len(_pick) >= 2: break
        return " ".join(_norm(x) for x in _pick)
    def extract_topic_tags(self, text, n=3):
        toks = [t for t in tokenize(text) if len(t) > 3]
        if not toks: return []
        bigrams = [f"{toks[i]} {toks[i+1]}" for i in range(len(toks)-1)]
        from collections import Counter
        return [b for b, _ in Counter(bigrams).most_common(n)]
    def judge(self, system, prompt):
        return {"score": 0.5, "rationale": "offline"}


_FACT_P = 'Extract atomic facts as JSON {"facts": [...]}.\n\nText:\n'
_SUM_P = 'Summarize in 1-2 sentences as JSON {"summary": "..."}.\n\nText:\n'


class OpenAICompatibleClient(LLMClient):
    backend = "openai-compatible"  # display-only override (base class default is "abstract")
    def __init__(self, config):
        self.config = config
        key = config.resolved_api_key()
        if not key: raise RuntimeError(f"API key missing. Set {config.api_key_env}.")
        from openai import OpenAI
        self.client = OpenAI(api_key=key, base_url=config.base_url, timeout=config.timeout_s)
    def chat_text(self, system, prompt):
        last = None
        for _ in range(self.config.max_retries):
            try:
                r = self.client.chat.completions.create(model=self.config.model,
                    messages=[{"role":"system","content":system},{"role":"user","content":prompt}])
                return r.choices[0].message.content or ""
            except Exception as e: last = e
        raise RuntimeError(f"LLM failed: {last}")
    def chat_json(self, system, prompt):
        for _ in range(self.config.max_retries):
            try:
                r = self.client.chat.completions.create(model=self.config.model,
                    response_format={"type":"json_object"},
                    messages=[{"role":"system","content":system},{"role":"user","content":prompt}])
                return _parse_json(r.choices[0].message.content or "{}")
            except Exception:
                try: return _parse_json(self.chat_text(system, prompt))
                except Exception: continue
        return {}
    def extract_facts(self, text):
        out = self.chat_json("Extract facts.", _FACT_P + text)
        fs = out.get("facts") if isinstance(out, dict) else None
        return [str(f) for f in fs] if isinstance(fs, list) and fs else []
    def summarize(self, text):
        out = self.chat_json("Summarize.", _SUM_P + text)
        return str(out.get("summary","")).strip() if isinstance(out, dict) else ""
    def extract_topic_tags(self, text, n=3):
        p = f'Return {n} tags as JSON {{"tags":[...]}}.\n\nText:\n' + text
        out = self.chat_json("Tag topics.", p)
        ts = out.get("tags") if isinstance(out, dict) else None
        return [str(t) for t in ts] if isinstance(ts, list) else []
    def judge(self, system, prompt):
        full = f"You are an LLM judge. {system}\nReturn JSON {{\"score\":float,\"rationale\":\"...\"}}.\n\n{prompt}"
        out = self.chat_json(system, full)
        if isinstance(out, dict) and "score" in out:
            try: return {"score": float(out["score"]), "rationale": str(out.get("rationale",""))}
            except (TypeError, ValueError): pass
        return {"score":0.5,"rationale":"unparsed"}


def make_client(config):
    if config.backend in {"offline","synthetic"}: return OfflineLLMClient()
    try: return OpenAICompatibleClient(config)
    except Exception: return OfflineLLMClient()


def _parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"): text = text[4:].strip()
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e >= s: text = text[s:e+1]
    return json.loads(text)
