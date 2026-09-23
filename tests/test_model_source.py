from pathlib import Path

import pytest

from src.model.load import resolve_model_source


def test_resolve_model_source_prefers_existing_local_path(tmp_path: Path) -> None:
    local = tmp_path / "model"
    local.mkdir()
    source, is_local = resolve_model_source(
        {"id": "remote/model", "local_path": str(local), "local_files_only": True}
    )
    assert source == str(local)
    assert is_local is True


def test_resolve_model_source_rejects_missing_required_local_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_model_source(
            {
                "id": "remote/model",
                "local_path": str(tmp_path / "missing"),
                "local_files_only": True,
            }
        )
