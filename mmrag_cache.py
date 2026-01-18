from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def _hash_payload(payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def cache_get(cache_dir: Path, namespace: str, payload: Dict[str, Any]) -> Dict[str, Any] | None:
    key = _hash_payload(payload)
    path = cache_dir / namespace / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def cache_set(cache_dir: Path, namespace: str, payload: Dict[str, Any], value: Dict[str, Any]) -> None:
    path = cache_dir / namespace
    path.mkdir(parents=True, exist_ok=True)
    key = _hash_payload(payload)
    out_path = path / f"{key}.json"
    out_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
