from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import cast

ENCODED_SKILL_DIR_PREFIX = "__xiaoman_plugin_skill__"
MANAGED_SKILL_COPY_MARKER = ".xiaoman-plugin-link.json"

_WINDOWS_INVALID_CHARS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def encode_plugin_skill_dir_name(logical_name: str) -> str:
    """Return a portable directory component for a logical plugin skill name."""
    if _is_portable_directory_component(logical_name):
        return logical_name
    encoded = base64.urlsafe_b64encode(logical_name.encode("utf-8")).decode("ascii")
    return f"{ENCODED_SKILL_DIR_PREFIX}{encoded.rstrip('=')}"


def decode_plugin_skill_dir_name(directory_name: str) -> str:
    """Decode names emitted by :func:`encode_plugin_skill_dir_name`."""
    if not directory_name.startswith(ENCODED_SKILL_DIR_PREFIX):
        return directory_name
    token = directory_name[len(ENCODED_SKILL_DIR_PREFIX) :]
    if not token:
        return directory_name
    padded = token + "=" * (-len(token) % 4)
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        return directory_name
    if encode_plugin_skill_dir_name(decoded) != directory_name:
        return directory_name
    return decoded


def plugin_skill_materialization_path(skills_dir: Path, logical_name: str) -> Path:
    return skills_dir / encode_plugin_skill_dir_name(logical_name)


def plugin_skill_logical_name(path: Path) -> str:
    """Return the public skill name represented by a workspace directory."""
    metadata = read_managed_skill_copy_marker(path)
    if metadata is not None:
        return metadata["logical_name"]
    if path.is_symlink():
        return decode_plugin_skill_dir_name(path.name)
    return path.name


def write_managed_skill_copy_marker(
    copied_dir: Path,
    *,
    logical_name: str,
    target: Path,
    fingerprint: str,
) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "logical_name": logical_name,
        "target": str(target.resolve(strict=False)),
        "fingerprint": fingerprint,
    }
    marker = copied_dir / MANAGED_SKILL_COPY_MARKER
    _ = marker.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_managed_skill_copy_marker(path: Path) -> dict[str, str] | None:
    marker = path / MANAGED_SKILL_COPY_MARKER
    if not marker.is_file():
        return None
    try:
        loaded = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    data = cast(dict[object, object], loaded)
    logical_name = str(data.get("logical_name") or "").strip()
    target = str(data.get("target") or "").strip()
    fingerprint = str(data.get("fingerprint") or "").strip()
    if not logical_name or not target or not fingerprint:
        return None
    return {
        "logical_name": logical_name,
        "target": target,
        "fingerprint": fingerprint,
    }


def is_plugin_skill_materialized(
    path: Path,
    *,
    logical_name: str | None = None,
) -> bool:
    if not (path / "SKILL.md").is_file():
        return False
    if path.is_symlink():
        return True
    metadata = read_managed_skill_copy_marker(path)
    if metadata is None:
        return False
    return logical_name is None or metadata["logical_name"] == logical_name


def _is_portable_directory_component(value: str) -> bool:
    if not value or value in {".", ".."} or value != value.strip():
        return False
    if value.startswith(ENCODED_SKILL_DIR_PREFIX):
        return False
    if value.endswith((" ", ".")):
        return False
    if any(ord(char) < 32 or char in _WINDOWS_INVALID_CHARS for char in value):
        return False
    stem = value.split(".", 1)[0].upper()
    return stem not in _WINDOWS_RESERVED_NAMES
