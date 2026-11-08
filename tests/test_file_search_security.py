"""Tests for file search security: path traversal, symlink escapes, allowlists, and size limits."""

import os
from pathlib import Path
import pytest

from nino.tools.files import FileReadTool, FileSearchTool


@pytest.fixture
def setup_files(tmp_path):
    # Allowed sandbox directory
    allowed_dir = tmp_path / "allowed_sandbox"
    allowed_dir.mkdir()

    # Secret directory outside allowed sandbox
    forbidden_dir = tmp_path / "forbidden_area"
    forbidden_dir.mkdir()

    # Files
    safe_file = allowed_dir / "safe_notes.txt"
    safe_file.write_text("Hello from safe note", encoding="utf-8")

    large_file = allowed_dir / "large_data.txt"
    large_file.write_text("X" * 1000, encoding="utf-8")

    secret_file = forbidden_dir / "system_secrets.txt"
    secret_file.write_text("SUPER_SECRET_DATA", encoding="utf-8")

    # Symlink inside sandbox pointing outside to forbidden_dir
    symlink_file = allowed_dir / "symlink_escape.txt"
    try:
        os.symlink(str(secret_file), str(symlink_file))
        symlinks_supported = True
    except (OSError, NotImplementedError):
        symlinks_supported = False

    search_tool = FileSearchTool(allowed_dirs_str=str(allowed_dir))
    read_tool = FileReadTool(allowed_dirs_str=str(allowed_dir))

    return allowed_dir, forbidden_dir, search_tool, read_tool, symlinks_supported


@pytest.mark.asyncio
async def test_safe_file_operations(setup_files):
    allowed_dir, _, search_tool, read_tool, _ = setup_files

    # 1. Search finding safe file
    search_res = await search_tool.run(query="safe")
    assert "safe_notes.txt" in search_res

    # 2. Read safe file
    safe_path = str(allowed_dir / "safe_notes.txt")
    read_res = await read_tool.run(path=safe_path)
    assert "Hello from safe note" in read_res
    assert 'taint="UNTRUSTED"' in read_res


@pytest.mark.asyncio
async def test_path_traversal_rejection(setup_files):
    allowed_dir, forbidden_dir, search_tool, read_tool, _ = setup_files

    # Relative path traversal attempts
    traversal_path = str(allowed_dir / ".." / "forbidden_area" / "system_secrets.txt")
    read_res = await read_tool.run(path=traversal_path)
    assert "Access Denied" in read_res
    assert "resolves outside allowed directories" in read_res

    # Classic UNIX traversal
    unix_traversal = "../../etc/passwd"
    read_res_unix = await read_tool.run(path=unix_traversal)
    assert "Access Denied" in read_res_unix

    # Windows root traversal
    win_traversal = r"C:\Windows\System32\drivers\etc\hosts"
    read_res_win = await read_tool.run(path=win_traversal)
    assert "Access Denied" in read_res_win

    # Search traversal attempt
    search_res = await search_tool.run(query="secrets", directory=str(forbidden_dir))
    assert "Access Denied" in search_res


@pytest.mark.asyncio
async def test_symlink_escape_rejection(setup_files):
    allowed_dir, _, _, read_tool, symlinks_supported = setup_files
    if not symlinks_supported:
        pytest.skip("Symlink creation not permitted or supported on this environment.")

    symlink_path = str(allowed_dir / "symlink_escape.txt")
    read_res = await read_tool.run(path=symlink_path)
    # The realpath of the symlink resolves into forbidden_area, so must be denied
    assert "Access Denied" in read_res
    assert "resolves outside allowed directories" in read_res


@pytest.mark.asyncio
async def test_file_size_limit_enforcement(setup_files):
    allowed_dir, _, _, read_tool, _ = setup_files

    large_path = str(allowed_dir / "large_data.txt")
    # Read with max_bytes = 100 on a 1000 byte file
    read_res = await read_tool.run(path=large_path, max_bytes=100)
    assert "Notice: File truncated, read 100 of 1000 bytes" in read_res
