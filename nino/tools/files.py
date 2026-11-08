"""File search and read tools strictly restricted to configured allowed directories."""

import os
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

from nino.tools.base import BaseTool
from nino.tools.permissions import PermissionTier
from nino.tools.taint import TaintTracker


def _is_path_allowed(target_path: Path, allowed_dirs: list[Path]) -> bool:
    """Check if the realpath of target_path is within any of the allowed directories."""
    try:
        resolved_target = target_path.resolve()
    except Exception:
        return False

    for allowed_dir in allowed_dirs:
        try:
            resolved_allowed = allowed_dir.resolve()
            # If target starts with allowed directory path
            if resolved_target == resolved_allowed or resolved_allowed in resolved_target.parents:
                return True
        except Exception:
            continue
    return False


class FileSearchInput(BaseModel):
    query: str = Field(..., description="File name pattern or substring to search for")
    directory: str | None = Field(default=None, description="Optional subdirectory within allowed roots")
    max_results: int = Field(default=10, description="Maximum number of file paths to return")


class FileSearchTool(BaseTool):
    name = "file_search"
    description = "Search for files within allowed project directories. Read-only."
    args_schema = FileSearchInput
    tier = PermissionTier.READ_ONLY

    def __init__(self, allowed_dirs_str: str = ".,test_notes,data"):
        raw_dirs = [d.strip() for d in allowed_dirs_str.split(",") if d.strip()]
        self.allowed_dirs = [Path(d).resolve() for d in raw_dirs]

    async def run(self, query: str, directory: str | None = None, max_results: int = 10) -> str:
        search_roots = self.allowed_dirs
        if directory:
            candidate = Path(directory).resolve()
            if not _is_path_allowed(candidate, self.allowed_dirs):
                return f"Access Denied: Specified directory '{directory}' is outside allowed directories."
            search_roots = [candidate]

        matches: list[str] = []
        lower_query = query.lower()

        for root_dir in search_roots:
            if not root_dir.exists():
                continue
            for dirpath, dirnames, filenames in os.walk(root_dir):
                # Filter out hidden/venv directories
                dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__" and d != "venv"]
                for f in filenames:
                    if lower_query in f.lower():
                        full_path = Path(dirpath) / f
                        if _is_path_allowed(full_path, self.allowed_dirs):
                            matches.append(str(full_path))
                            if len(matches) >= max_results:
                                break
                if len(matches) >= max_results:
                    break

        if not matches:
            return f"No files found matching query '{query}' in allowed directories."

        lines = [f"Found {len(matches)} matching file(s):"]
        for m in matches:
            lines.append(f"- {m}")
        return "\n".join(lines)


class FileReadInput(BaseModel):
    path: str = Field(..., description="Relative or absolute file path to read")
    max_bytes: int = Field(default=50000, description="Maximum bytes to read from file")


class FileReadTool(BaseTool):
    name = "file_read"
    description = "Read the contents of a file within allowed directories. Read-only."
    args_schema = FileReadInput
    tier = PermissionTier.READ_ONLY

    def __init__(self, allowed_dirs_str: str = ".,test_notes,data"):
        raw_dirs = [d.strip() for d in allowed_dirs_str.split(",") if d.strip()]
        self.allowed_dirs = [Path(d).resolve() for d in raw_dirs]

    async def run(self, path: str, max_bytes: int = 50000) -> str:
        target = Path(path)
        if not target.is_absolute():
            target = (Path.cwd() / target).resolve()
        else:
            target = target.resolve()

        if not _is_path_allowed(target, self.allowed_dirs):
            return f"Access Denied: File path '{path}' resolves outside allowed directories."

        if not target.exists():
            return f"File '{path}' does not exist."

        if not target.is_file():
            return f"Path '{path}' is not a regular file."

        file_size = target.stat().st_size
        try:
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_bytes)
        except Exception as e:
            return f"Error reading file '{path}': {e}"

        truncated_notice = ""
        if file_size > max_bytes:
            truncated_notice = f"\n[Notice: File truncated, read {max_bytes} of {file_size} bytes]"

        raw_output = f"File: {target}\nSize: {file_size} bytes\n\n{content}{truncated_notice}"
        return TaintTracker.wrap_untrusted(raw_output, source=str(target))
