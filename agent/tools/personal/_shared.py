from __future__ import annotations

import json
from typing import Any


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)
