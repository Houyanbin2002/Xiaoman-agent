from __future__ import annotations

from typing import TypeAlias


# SQLite、向量召回、关键词召回和注入治理之间的统一边界对象。
MemoryHit: TypeAlias = dict[str, object]
