from pathlib import Path

import pytest

from lc_agent.tools.file import FileToolError, _list_directory


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "todo.md").write_text("hello")
    (tmp_path / "readme.md").write_text("root file")
    return tmp_path


def test_list_directory_lists_immediate_entries(workspace: Path) -> None:
    assert _list_directory(".", workspace_root=workspace) == ["notes/", "readme.md"]
    assert _list_directory("notes", workspace_root=workspace) == ["todo.md"]


def test_list_directory_rejects_missing_path(workspace: Path) -> None:
    with pytest.raises(FileToolError, match="directory not found"):
        _list_directory("missing", workspace_root=workspace)


def test_list_directory_rejects_file_path(workspace: Path) -> None:
    with pytest.raises(FileToolError, match="not a directory"):
        _list_directory("readme.md", workspace_root=workspace)


def test_list_directory_rejects_path_escape(workspace: Path) -> None:
    with pytest.raises(FileToolError, match="path escapes workspace"):
        _list_directory("../outside", workspace_root=workspace)
