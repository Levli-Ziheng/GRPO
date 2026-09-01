from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.common import load_json, sha256_file
from src.utils.config import load_yaml, write_json


def build_manifest(config: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]
    required = {
        "train_json": Path(source["train_json"]),
        "dev_json": Path(source["dev_json"]),
        "tables_json": Path(source["tables_json"]),
        "database_dir": Path(source["database_dir"]),
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CSpider source assets: {missing}")

    train = load_json(required["train_json"])
    dev = load_json(required["dev_json"])
    tables = load_json(required["tables_json"])
    if not all(isinstance(value, list) for value in (train, dev, tables)):
        raise ValueError("train.json, dev.json and tables.json must contain JSON arrays")

    table_db_ids = {item["db_id"] for item in tables}
    train_db_ids = {item["db_id"] for item in train}
    dev_db_ids = {item["db_id"] for item in dev}
    database_files: dict[str, dict[str, Any]] = {}
    missing_databases: list[str] = []
    for db_id in sorted(table_db_ids):
        path = required["database_dir"] / db_id / f"{db_id}.sqlite"
        if not path.is_file():
            missing_databases.append(str(path))
            continue
        database_files[db_id] = {
            "path": path.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    if missing_databases:
        raise FileNotFoundError(
            f"Missing {len(missing_databases)} databases, first entries: {missing_databases[:5]}"
        )
    unknown = sorted((train_db_ids | dev_db_ids) - table_db_ids)
    if unknown:
        raise ValueError(f"Samples reference db_id values absent from tables.json: {unknown}")

    files = {}
    for name in ("train_json", "dev_json", "tables_json"):
        path = required[name]
        files[name] = {
            "path": path.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    return {
        "source": source["name"],
        "source_version": source["version"],
        "homepage": source["homepage"],
        "repository": source["repository"],
        "license_note": source["license_note"],
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "databases": database_files,
        "counts": {
            "train_samples": len(train),
            "dev_samples": len(dev),
            "table_entries": len(tables),
            "database_files": len(database_files),
            "train_databases": len(train_db_ids),
            "dev_databases": len(dev_db_ids),
        },
        "official_train_dev_db_overlap": sorted(train_db_ids & dev_db_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit locally imported CSpider assets and record hashes."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    report = build_manifest(config)
    output = Path(config["paths"]["source_manifest"])
    write_json(output, report)
    print(
        f"CSpider source OK: {report['counts']['train_samples']} train, "
        f"{report['counts']['dev_samples']} dev, "
        f"{report['counts']['database_files']} databases"
    )
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
