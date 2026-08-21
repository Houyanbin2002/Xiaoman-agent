from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import zipfile


APP_EXCLUDED_DIRS = {
    ".codex-commander-prototype",
    ".codex-runtime",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".runtime",
    ".venv",
    "__pycache__",
    "logs",
    "node_modules",
    "release",
}
GIT_EXCLUDED_FILE_NAMES = {
    "index.lock",
    "shallow.lock",
}
APP_EXCLUDED_FILE_NAMES = {
    "config.toml",
}
APP_EXCLUDED_SUFFIXES = {
    ".log",
    ".pyc",
    ".pyo",
}
PRIVATE_TRANSIENT_SUFFIXES = {
    ".db-journal",
    ".db-shm",
    ".db-wal",
    ".lock",
    ".pid",
    ".sock",
}


def _now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _should_copy_app(relative: Path) -> bool:
    if any(part in APP_EXCLUDED_DIRS for part in relative.parts[:-1]):
        return False
    if relative.name in APP_EXCLUDED_FILE_NAMES:
        return False
    if relative.name == "config.local.toml":
        return relative.as_posix() == "plugins/default_memory/config.local.toml"
    if relative.name == ".env":
        return False
    return relative.suffix.lower() not in APP_EXCLUDED_SUFFIXES


def _copy_app(source_root: Path, target_root: Path) -> int:
    count = 0
    for source in source_root.rglob("*"):
        relative = source.relative_to(source_root)
        if source.is_dir() or not _should_copy_app(relative):
            continue
        _copy_file(source, target_root / relative)
        count += 1
    return count


def _copy_git_metadata(source_root: Path, target_root: Path) -> int:
    """Copy local Git history so the migrated tree remains a real checkout."""
    source_git = source_root / ".git"
    target_git = target_root / ".git"
    if not source_git.is_dir():
        return 0
    count = 0
    for source in source_git.rglob("*"):
        if source.is_dir() or source.name in GIT_EXCLUDED_FILE_NAMES:
            continue
        _copy_file(source, target_git / source.relative_to(source_git))
        count += 1
    return count


def _create_gitlink_directories(source_root: Path, target_root: Path) -> int:
    """Preserve uninitialized submodule directories in the portable checkout."""
    try:
        output = subprocess.check_output(
            ["git", "ls-files", "--stage"],
            cwd=source_root,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return 0
    count = 0
    for line in output.splitlines():
        metadata, _, relative = line.partition("\t")
        mode = metadata.split(" ", 1)[0]
        if mode != "160000" or not relative:
            continue
        (target_root / Path(relative)).mkdir(parents=True, exist_ok=True)
        count += 1
    return count


def _backup_sqlite(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_uri = source.resolve().as_uri() + "?mode=ro"
    source_db = sqlite3.connect(source_uri, uri=True, timeout=30.0)
    target_db = sqlite3.connect(target)
    try:
        source_db.execute("PRAGMA busy_timeout = 30000")
        source_db.backup(target_db)
    finally:
        target_db.close()
        source_db.close()


def _copy_private_tree(
    source_root: Path,
    target_root: Path,
    *,
    skip_zip_files: bool = False,
) -> tuple[int, int]:
    files = 0
    databases = 0
    if not source_root.exists():
        return files, databases
    for source in source_root.rglob("*"):
        if source.is_dir():
            continue
        relative = source.relative_to(source_root)
        suffix = source.suffix.lower()
        if any(source.name.lower().endswith(item) for item in PRIVATE_TRANSIENT_SUFFIXES):
            continue
        if skip_zip_files and suffix == ".zip":
            continue
        target = target_root / relative
        if suffix == ".db":
            _backup_sqlite(source, target)
            databases += 1
        else:
            _copy_file(source, target)
        files += 1
    return files, databases


def _git_metadata(source_root: Path) -> tuple[str, bool, str]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            text=True,
            encoding="utf-8",
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=source_root,
            text=True,
            encoding="utf-8",
        )
        return head, bool(status.strip()), status
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True, "Git metadata unavailable.\n"


def _write_zip(source_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for source in sorted(source_dir.rglob("*")):
            if source.is_file():
                archive.write(source, source.relative_to(source_dir.parent))
            elif source.is_dir() and not any(source.iterdir()):
                relative = source.relative_to(source_dir.parent).as_posix().rstrip("/") + "/"
                archive.writestr(relative, "")


def build_bundle(source_root: Path, output_root: Path) -> tuple[Path, Path]:
    stamp = _now_stamp()
    bundle_name = f"Xiaoman-Agent-Windows-x64-{stamp}"
    bundle_root = output_root / bundle_name
    archive_path = output_root / f"{bundle_name}.zip"
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True)

    app_root = bundle_root / "app"
    migration_root = bundle_root / "migration"
    app_files = _copy_app(source_root, app_root)
    git_files = _copy_git_metadata(source_root, app_root)
    gitlinks = _create_gitlink_directories(source_root, app_root)

    packaging_root = source_root / "packaging" / "windows"
    for name in (
        "install.ps1",
        "start.ps1",
        "setup-development.ps1",
        "install.cmd",
        "start.cmd",
        "setup-development.cmd",
    ):
        _copy_file(packaging_root / name, bundle_root / name)
    _copy_file(
        packaging_root / "README-MIGRATION.md",
        bundle_root / "README-MIGRATION.md",
    )

    private_files = 0
    databases = 0
    user_home = Path.home()
    xiaoman_home = user_home / ".xiaoman"
    for name in ("workspace", "skills", "marketplace"):
        copied, backed_up = _copy_private_tree(
            xiaoman_home / name,
            migration_root / "xiaoman" / name,
        )
        private_files += copied
        databases += backed_up

    # MarkItDown's virtual environment is machine/path specific and is rebuilt by
    # install.ps1. Portable native MCP binaries can be copied as-is.
    copied, backed_up = _copy_private_tree(
        xiaoman_home / "mcp" / "xiaohongshu",
        migration_root / "xiaoman" / "mcp" / "xiaohongshu",
        skip_zip_files=True,
    )
    private_files += copied
    databases += backed_up

    plugin_home = user_home / ".xiaoman-plugin"
    copied, backed_up = _copy_private_tree(
        plugin_home,
        migration_root / "xiaoman-plugin",
    )
    private_files += copied
    databases += backed_up

    config = source_root / "config.toml"
    if config.is_file():
        _copy_file(config, migration_root / "config.toml")
        private_files += 1
    for local_config in source_root.rglob("config.local.toml"):
        relative = local_config.relative_to(source_root)
        if any(part in APP_EXCLUDED_DIRS for part in relative.parts):
            continue
        _copy_file(
            local_config,
            migration_root / "app-overrides" / relative,
        )
        private_files += 1

    git_head, git_dirty, git_status = _git_metadata(source_root)
    (bundle_root / "WORKTREE_STATUS.txt").write_text(
        git_status,
        encoding="utf-8",
    )
    manifest = {
        "bundle_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": "windows-x64",
        "python": "3.12",
        "source_root": str(source_root),
        "source_user_home": str(user_home),
        "git_head": git_head,
        "git_dirty": git_dirty,
        "contains_private_data": True,
        "app_file_count": app_files,
        "git_file_count": git_files,
        "gitlink_count": gitlinks,
        "private_file_count": private_files,
        "sqlite_backup_count": databases,
        "credentials_note": (
            "config.toml is included; Windows Credential Manager entries are not. "
            "OAuth and QR-login integrations must be authorized again."
        ),
    }
    (bundle_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _write_zip(bundle_root, archive_path)
    return bundle_root, archive_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a private Xiaoman migration bundle")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "release",
    )
    args = parser.parse_args()
    source_root = args.source.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    bundle_root, archive_path = build_bundle(source_root, output_root)
    print(f"bundle_dir={bundle_root}")
    print(f"archive={archive_path}")
    print(f"archive_bytes={archive_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
