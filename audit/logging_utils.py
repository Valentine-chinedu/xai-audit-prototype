import json
import os
from datetime import datetime

def ensure_outputs_dir(path="outputs"):
    os.makedirs(path, exist_ok=True)

def append_jsonl(record: dict, filepath="outputs/runs.jsonl"):
    ensure_outputs_dir(os.path.dirname(filepath) or "outputs")
    record = dict(record)
    record["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
