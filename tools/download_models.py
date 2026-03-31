"""Download public Hugging Face model snapshots for Quench benchmarks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download public model snapshots for Quench benchmarks.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory used as the Hugging Face cache root for downloaded snapshots",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec in the form repo_id or repo_id@revision",
    )
    parser.add_argument(
        "--manifest",
        help="Optional output path for the download manifest JSON",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[
            "*.safetensors",
            "*.safetensors.index.json",
            "config.json",
            "generation_config.json",
        ],
        help="Allow-pattern forwarded to snapshot_download; may be repeated",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        from huggingface_hub import HfApi, snapshot_download
    except Exception as exc:
        print(
            "download_models.py requires huggingface_hub. "
            f"Import failed with {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "model-download-manifest.json"
    manifest_dir = manifest_path.parent.resolve()
    token = os.environ.get("HF_TOKEN") or None

    api = HfApi(token=token)
    models: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for model_spec in args.model:
        repo_id, requested_revision = _parse_model_spec(model_spec)
        try:
            info = api.model_info(repo_id=repo_id, revision=requested_revision)
            resolved_revision = str(info.sha)
            local_path = Path(
                snapshot_download(
                    repo_id=repo_id,
                    revision=resolved_revision,
                    token=token,
                    cache_dir=str(output_dir),
                    allow_patterns=args.allow_pattern,
                )
            )
            files = _collect_file_metadata(local_path)
            total_bytes = sum(int(item["size_bytes"]) for item in files)
            local_path_resolved = local_path.resolve()
            try:
                manifest_local_path = str(local_path_resolved.relative_to(manifest_dir))
            except ValueError:
                manifest_local_path = os.path.relpath(local_path_resolved, manifest_dir)
            models.append(
                {
                    "repo_id": repo_id,
                    "requested_revision": requested_revision,
                    "resolved_revision": resolved_revision,
                    "local_path": manifest_local_path,
                    "total_downloaded_bytes": total_bytes,
                    "files": files,
                }
            )
            print(
                f"downloaded {repo_id}@{resolved_revision} to {local_path} "
                f"({total_bytes} bytes, {len(files)} files)"
            )
        except Exception as exc:
            failures.append(
                {
                    "repo_id": repo_id,
                    "requested_revision": requested_revision,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"download failed for {repo_id}@{requested_revision}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "allow_patterns": args.allow_pattern,
        "models": models,
        "failures": failures,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote manifest {manifest_path}")

    if not models:
        sys.exit(1)


def _collect_file_metadata(local_path: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for file_path in sorted(path for path in local_path.rglob("*") if path.is_file()):
        kind = "weight" if file_path.suffix == ".safetensors" else "metadata"
        files.append(
            {
                "path": str(file_path.relative_to(local_path)),
                "kind": kind,
                "size_bytes": file_path.stat().st_size,
                "sha256": (_sha256(file_path) if kind == "weight" else ""),
            }
        )
    return files


def _parse_model_spec(model_spec: str) -> tuple[str, str]:
    if "@" not in model_spec:
        return model_spec, "main"
    repo_id, requested_revision = model_spec.rsplit("@", 1)
    return repo_id, requested_revision or "main"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
