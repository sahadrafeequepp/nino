"""Sandboxed Python code execution tool in isolated Docker containers with strict security constraints."""

import asyncio
import shutil
import subprocess
from typing import Any
from pydantic import BaseModel, Field

from nino.tools.base import BaseTool
from nino.tools.permissions import PermissionTier
from nino.tools.taint import TaintTracker


class CodeExecutionInput(BaseModel):
    code: str = Field(..., description="Python 3 script/snippet to execute in the isolated sandbox")


class PythonCodeExecutionTool(BaseTool):
    name = "code_execution"
    description = (
        "Execute a Python script inside an isolated, non-networked Docker container. "
        "SENSITIVE tier: requires explicit interactive confirmation every time."
    )
    args_schema = CodeExecutionInput
    tier = PermissionTier.SENSITIVE

    def __init__(
        self,
        enabled: bool = False,
        timeout_seconds: float = 10.0,
        docker_image: str = "python:3.11-slim",
    ):
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.docker_image = docker_image

    async def run(self, code: str) -> str:
        if not self.enabled:
            return (
                "Code execution is currently disabled (ENABLE_CODE_EXECUTION=false). "
                "To enable, set ENABLE_CODE_EXECUTION=true and ensure Docker is installed and running."
            )

        docker_path = shutil.which("docker")
        if not docker_path:
            return "Docker executable not found on system PATH. Ensure Docker Desktop/daemon is installed."

        # Verify docker daemon accessibility
        docker_cmd = [
            docker_path,
            "run",
            "--rm",
            "-i",
            "--network", "none",
            "--read-only",
            "--user", "1000:1000",
            "--cpus", "1.0",
            "--memory", "256m",
            "--pids-limit", "64",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            self.docker_image,
            "python",
            "-c",
            code,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout_seconds,
            )
            stdout_str = stdout.decode("utf-8", errors="replace").strip()
            stderr_str = stderr.decode("utf-8", errors="replace").strip()
            exit_code = process.returncode

            output_lines = [f"Exit code: {exit_code}"]
            if stdout_str:
                output_lines.append(f"STDOUT:\n{stdout_str}")
            if stderr_str:
                output_lines.append(f"STDERR:\n{stderr_str}")

            raw_result = "\n\n".join(output_lines)
            return TaintTracker.wrap_untrusted(raw_result, source="docker_code_execution")

        except asyncio.TimeoutError:
            try:
                process.kill()
            except Exception:
                pass
            return f"Execution timed out after {self.timeout_seconds} seconds."
        except Exception as e:
            return f"Failed to execute code in Docker container: {e}"
