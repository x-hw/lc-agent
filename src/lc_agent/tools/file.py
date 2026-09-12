from pathlib import Path

from langchain.tools import tool


def make_list_directory_tool(workspace_root: Path):

    @tool
    def list_directory(path: str = ".") -> str:
        """List immediate entries in a workspace directory."""
        entries = _list_directory(path, workspace_root)
        return "\n".join(entries)

    return list_directory


def make_read_text_file_tool(workspace_root: Path, max_chars: int = 5000):

    @tool
    def read_text_file(path: str) -> str:
        """Read a text file in a workspace directory."""
        return _read_text_file(path, workspace_root, max_chars)

    return read_text_file


class FileToolError(ValueError):
    pass


def _safe_resolve(path: str, workspace_root: Path) -> Path:
    resolved = (workspace_root / path).resolve()
    if not resolved.is_relative_to(workspace_root.resolve()):
        raise FileToolError(f"path escapes workspace: {path}")
    return resolved


def _list_directory(
    path: str,
    workspace_root: Path,
) -> list[str]:
    resolved = _safe_resolve(path, workspace_root)
    if not resolved.exists():
        raise FileToolError(f"directory not found: {path}")
    if not resolved.is_dir():
        raise FileToolError(f"not a directory: {path}")

    entries = []
    for entry in sorted(resolved.iterdir(), key=lambda item: item.name):
        if entry.is_dir():
            entries.append(f"{entry.name}/")
        else:
            entries.append(entry.name)
    return entries


def _read_text_file(path: str, workspace_root: Path, max_chars: int) -> str:
    resolved = _safe_resolve(path, workspace_root)
    if not resolved.exists():
        raise FileToolError(f"file not found: {path}")
    if not resolved.is_file():
        raise FileToolError(f"not a file: {path}")
    with resolved.open() as fp:
        return fp.read(max_chars)
