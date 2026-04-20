"""Produce a new live manifest by grafting the Nitro-E entry from the local
draft onto the current live models.json. Other draft-only models (adaface,
yolov9s, efficientad, basicpitch) are intentionally NOT touched here so we
don't surprise users with half-staged models.
"""

import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_BACKUP = os.environ.get("LIVE",
    os.path.join(ROOT, ".manifest_backups", "models.json.20260420-131546"))
DRAFT = os.path.join(ROOT, "sample_apps", "CoreMLModelsApp", "models_draft.json")
OUT = os.path.join(ROOT, ".manifest_backups", "models.nitroe_live.json")


def main() -> None:
    with open(LIVE_BACKUP, encoding="utf-8") as f:
        live = json.load(f)
    with open(DRAFT, encoding="utf-8") as f:
        draft = json.load(f)

    nitroe = next((m for m in draft["models"] if m["id"] == "nitroe"), None)
    if nitroe is None:
        sys.exit("nitroe entry missing from draft")

    # Insert Nitro-E before the first hypersd entry (keeps the Generation
    # section ordering: Nitro-E, then Hyper-SD).
    models = list(live["models"])
    insert_at = next((i for i, m in enumerate(models) if m["id"] == "hypersd"),
                     len(models))
    models.insert(insert_at, nitroe)

    new = dict(live)
    new["models"] = models
    new["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(new, f, ensure_ascii=False, indent=2)

    print(f"[live] {len(live['models'])} -> [new] {len(new['models'])} models")
    print(f"[wrote] {OUT}")
    print(f"[updated_at] {new['updated_at']}")


if __name__ == "__main__":
    main()
