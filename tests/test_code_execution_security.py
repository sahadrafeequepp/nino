"""Tests for sandboxed code execution security parameters and mock execution."""

from unittest.mock import AsyncMock, patch
import pytest

from nino.tools.code_execution import PythonCodeExecutionTool
from nino.tools.permissions import PermissionTier


def test_code_execution_tier_and_default_disabled():
    tool = PythonCodeExecutionTool(enabled=False)
    # Tier must be SENSITIVE (requires human confirmation)
    assert tool.tier == PermissionTier.SENSITIVE
    assert tool.name == "code_execution"


@pytest.mark.asyncio
async def test_code_execution_blocks_when_disabled():
    tool = PythonCodeExecutionTool(enabled=False)
    result = await tool.run(code="print('hello')")
    assert "currently disabled" in result


@pytest.mark.asyncio
async def test_docker_command_contains_all_security_constraints():
    tool = PythonCodeExecutionTool(enabled=True, timeout_seconds=5.0)

    # Mock shutil.which to pretend docker is present
    # and mock create_subprocess_exec to capture args
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"mock_output", b"")
    mock_proc.returncode = 0

    with patch("shutil.which", return_value="docker"):
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            res = await tool.run(code="print(1 + 1)")
            assert mock_exec.called
            cmd_args = list(mock_exec.call_args[0])

            # Verify all mandatory security containment flags
            assert "--network" in cmd_args and cmd_args[cmd_args.index("--network") + 1] == "none"
            assert "--read-only" in cmd_args
            assert "--user" in cmd_args and cmd_args[cmd_args.index("--user") + 1] == "1000:1000"
            assert "--cpus" in cmd_args and cmd_args[cmd_args.index("--cpus") + 1] == "1.0"
            assert "--memory" in cmd_args and cmd_args[cmd_args.index("--memory") + 1] == "256m"
            assert "--pids-limit" in cmd_args and cmd_args[cmd_args.index("--pids-limit") + 1] == "64"
            assert "--cap-drop" in cmd_args and cmd_args[cmd_args.index("--cap-drop") + 1] == "ALL"
            assert "--security-opt" in cmd_args and cmd_args[cmd_args.index("--security-opt") + 1] == "no-new-privileges"
            # Verify NO host mounts (-v or --volume or --mount)
            assert "-v" not in cmd_args
            assert "--volume" not in cmd_args
            assert "--mount" not in cmd_args

            # Verify output is wrapped with untrusted taint tags
            assert 'taint="UNTRUSTED"' in res
            assert "mock_output" in res
