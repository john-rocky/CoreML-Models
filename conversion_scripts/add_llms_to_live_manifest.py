"""Graft the new CoreML-LLM entries (Gemma 4 E4B, Qwen3.5 2B, Qwen3.5 0.8B,
Qwen3-VL 2B) plus the refreshed Gemma 4 E2B description onto the current live
manifest. Writes a new local backup — DOES NOT push to HuggingFace.

Usage:
    python conversion_scripts/add_llms_to_live_manifest.py

Follows the same pattern as add_nitroe_to_live_manifest.py: takes LIVE (a
backup snapshot of the HF-hosted models.json) and DRAFT (the local scratchpad),
copies the `llm` category + every llm-category entry from the draft onto the
live snapshot, writes to .manifest_backups/models.llm_update_live.json. The
user can then upload that file to HF when ready.
"""

import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_BACKUP = os.environ.get(
    "LIVE", os.path.join(ROOT, ".manifest_backups", "models.nitroe_live.json")
)
DRAFT = os.path.join(ROOT, "sample_apps", "CoreMLModelsApp", "models_draft.json")
OUT = os.path.join(ROOT, ".manifest_backups", "models.llm_update_live.json")


def main() -> None:
    with open(LIVE_BACKUP, encoding="utf-8") as f:
        live = json.load(f)
    with open(DRAFT, encoding="utf-8") as f:
        draft = json.load(f)

    draft_llms = [m for m in draft["models"] if m.get("category_id") == "llm"]
    if not draft_llms:
        sys.exit("no llm-category models found in draft")

    # Ensure the `llm` category is present in live (order=0 so LLMs appear at
    # the top of the Hub App home screen).
    live_cats = list(live.get("categories", []))
    if not any(c["id"] == "llm" for c in live_cats):
        llm_cat = next((c for c in draft["categories"] if c["id"] == "llm"), None)
        if llm_cat is None:
            sys.exit("draft has llm models but no llm category")
        live_cats.insert(0, llm_cat)

    # Replace every existing llm model in live with the draft version, keep
    # insertion order of the draft, then keep the rest of live as-is.
    llm_ids = {m["id"] for m in draft_llms}
    non_llm = [m for m in live["models"] if m["id"] not in llm_ids]
    models = draft_llms + non_llm

    new = dict(live)
    new["categories"] = live_cats
    new["models"] = models
    new["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(new, f, ensure_ascii=False, indent=2)

    print(f"[live] {len(live['models'])} -> [new] {len(new['models'])} models")
    print(f"[llm ] {[m['id'] for m in draft_llms]}")
    print(f"[wrote] {OUT}")
    print(f"[updated_at] {new['updated_at']}")


if __name__ == "__main__":
    main()
