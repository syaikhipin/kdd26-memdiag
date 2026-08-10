# Local LLM endpoint config

`kdd26_memdiag_config.json` lives here. It holds the user-specific OpenAI-compatible
endpoint used by the tutorial notebooks and the `run.py` CLI:

```json
{"base_url": "https://api.openai.com/v1", "api_key": "sk-...", "model": "gpt-4o"}
```

- **Gitignored** — it contains your API key and is never committed.
- Created automatically on first run by `setup_llm()` (notebooks) / `resolve_endpoint()`
  (CLI) in `source/config.py`; you can also edit it by hand.
- On Google Colab the canonical copy is on Drive
  (`/content/drive/MyDrive/kdd26_memdiag_config.json`); this file is the local fallback.
