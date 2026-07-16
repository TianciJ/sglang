#!/usr/bin/env python3
"""Copy a PD Flip artifact tree and redact credential-bearing log text."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path


SHANGHAI = timezone(timedelta(hours=8))
RULES = [
    (
        "python_repr_admin_api_key",
        re.compile(r"(?i)(admin_api_key\s*=\s*')[^']*(')"),
        r"\1[REDACTED]\2",
    ),
    (
        "environment_admin_api_key",
        re.compile(r"(?i)(ADMIN_API_KEY=)[^\s'\"]+"),
        r"\1[REDACTED]",
    ),
    (
        "authorization_bearer",
        re.compile(r"(?i)(Authorization:\s*Bearer\s+)[A-Za-z0-9._~+\-/=]+"),
        r"\1[REDACTED]",
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    shutil.copytree(source, destination)

    changes = []
    for path in sorted(p for p in destination.rglob("*") if p.is_file()):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        original = text
        applied = {}
        for name, pattern, replacement in RULES:
            text, count = pattern.subn(replacement, text)
            if count:
                applied[name] = count
        if text != original:
            path.write_text(text, encoding="utf-8")
            relative = path.relative_to(destination).as_posix()
            changes.append(
                {
                    "file": relative,
                    "rules": applied,
                    "source_sha256": sha256(source / relative),
                    "redacted_sha256": sha256(path),
                }
            )

    residuals = []
    for path in sorted(p for p in destination.rglob("*") if p.is_file()):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern, _replacement in RULES:
            if any("[REDACTED]" not in match.group(0) for match in pattern.finditer(text)):
                residuals.append({"file": path.relative_to(destination).as_posix(), "rule": name})

    manifest = {
        "source_directory": str(source),
        "redacted_directory": str(destination),
        "generated_at_shanghai": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
        "rules": [name for name, _pattern, _replacement in RULES],
        "changed_files": changes,
        "residual_match_count": len(residuals),
        "residual_matches": residuals,
        "note": "Only credential-bearing text was replaced; timing, request, metrics, and status data are unchanged.",
    }
    manifest_path = destination / "REDACTION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    archive = args.archive.resolve() if args.archive else destination.with_suffix(".tar.gz")
    if archive.name.endswith(".tar.gz"):
        archive_base = archive.with_suffix("").with_suffix("")
    else:
        archive_base = archive
    produced = Path(
        shutil.make_archive(
            str(archive_base),
            "gztar",
            root_dir=destination.parent,
            base_dir=destination.name,
        )
    )
    archive_manifest = {
        "archive": str(produced),
        "bytes": produced.stat().st_size,
        "sha256": sha256(produced),
        "redaction_manifest": str(manifest_path),
    }
    (destination / "REDACTED_ARCHIVE.json").write_text(
        json.dumps(archive_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(archive_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
