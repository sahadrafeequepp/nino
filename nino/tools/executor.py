"""Central tool executor enforcing permission tiers, taint escalation, timeouts, and audit logging."""

import asyncio
import re
import time
from typing import Any
from pydantic import BaseModel, ValidationError

from nino.security.audit import AuditLogger
from nino.security.privacy_gateway import SECRET_PATTERNS
from nino.tools.base import BaseTool
from nino.tools.confirmation import Confirmation, ConfirmationManager
from nino.tools.permissions import PermissionTier
from nino.tools.registry import ToolRegistry
from nino.tools.taint import TaintTracker


class ToolExecutionResult(BaseModel):
    """Result returned by the CentralExecutor."""

    success: bool
    tool_name: str
    output: Any | None = None
    error: str | None = None
    requires_confirmation: bool = False
    confirmation: Confirmation | None = None
    tier: str
    decision: str  # "executed", "confirmation_required", "blocked", "failed", "timeout"
    duration_ms: float = 0.0


class CentralExecutor:
    """The central and exclusive execution authority for all Nino tools."""

    def __init__(
        self,
        registry: ToolRegistry,
        confirmation_manager: ConfirmationManager,
        audit_logger: AuditLogger,
        allow_restricted_tools: bool = False,
        tool_timeout_seconds: float = 30.0,
        output_max_chars: int = 4000,
        log_content: bool = False,
    ):
        self.registry = registry
        self.confirmation_manager = confirmation_manager
        self.audit_logger = audit_logger
        self.allow_restricted_tools = allow_restricted_tools
        self.tool_timeout_seconds = tool_timeout_seconds
        self.output_max_chars = output_max_chars
        self.log_content = log_content

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: str | list[dict[str, Any]] | None = None,
        confirmation_id: str | None = None,
    ) -> ToolExecutionResult:
        """Execute a tool through policy enforcement, schema validation, and audit recording."""
        start_time = time.perf_counter()

        tool = self.registry.get(tool_name)
        if not tool:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._audit(tool_name, "UNKNOWN", "rejected_unknown_tool", duration_ms)
            return ToolExecutionResult(
                success=False,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' is not registered.",
                tier="UNKNOWN",
                decision="failed",
                duration_ms=duration_ms,
            )

        tier = tool.tier

        # 1. Validate arguments against Pydantic schema
        try:
            validated_args = tool.args_schema(**arguments).model_dump()
        except ValidationError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._audit(tool_name, tier.value, "schema_validation_failed", duration_ms)
            return ToolExecutionResult(
                success=False,
                tool_name=tool_name,
                error=f"Invalid arguments for tool '{tool_name}': {e}",
                tier=tier.value,
                decision="failed",
                duration_ms=duration_ms,
            )

        # 2. Check RESTRICTED tier policy
        if tier == PermissionTier.RESTRICTED and not self.allow_restricted_tools:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._audit(tool_name, tier.value, "blocked_restricted_disabled", duration_ms)
            return ToolExecutionResult(
                success=False,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' belongs to the RESTRICTED tier which is currently disabled in configuration.",
                tier=tier.value,
                decision="blocked",
                duration_ms=duration_ms,
            )

        # 3. Check Taint Escalation & Confirmation Requirements
        is_tainted = TaintTracker.is_tainted(context) if context else False
        needs_confirmation = False
        summary_reason = ""

        if tier == PermissionTier.READ_ONLY:
            needs_confirmation = False
        elif tier == PermissionTier.REVERSIBLE:
            if is_tainted:
                needs_confirmation = True
                summary_reason = f"[TAINT ESCALATION] Untrusted content in context; confirming {tool_name}"
            else:
                needs_confirmation = False
        elif tier in (PermissionTier.SENSITIVE, PermissionTier.RESTRICTED):
            needs_confirmation = True
            summary_reason = f"[{tier.value}] Action requires explicit confirmation: {tool_name}"

        # 4. Enforce Confirmation Flow
        if needs_confirmation:
            if not confirmation_id:
                # Generate pending confirmation and return without executing
                summary = self._generate_summary(tool, validated_args, summary_reason)
                conf = self.confirmation_manager.create_confirmation(
                    tool_name=tool_name,
                    arguments=validated_args,
                    summary=summary,
                    tier=tier,
                )
                duration_ms = (time.perf_counter() - start_time) * 1000
                self._audit(tool_name, tier.value, "confirmation_required", duration_ms)
                return ToolExecutionResult(
                    success=False,
                    tool_name=tool_name,
                    requires_confirmation=True,
                    confirmation=conf,
                    error=f"Action requires confirmation: {summary}",
                    tier=tier.value,
                    decision="confirmation_required",
                    duration_ms=duration_ms,
                )

            # Validate provided confirmation token
            valid, err_msg = self.confirmation_manager.validate_and_consume(
                confirmation_id=confirmation_id,
                tool_name=tool_name,
                arguments=validated_args,
            )
            if not valid:
                duration_ms = (time.perf_counter() - start_time) * 1000
                self._audit(tool_name, tier.value, "confirmation_rejected", duration_ms)
                return ToolExecutionResult(
                    success=False,
                    tool_name=tool_name,
                    error=f"Confirmation error: {err_msg}",
                    tier=tier.value,
                    decision="blocked",
                    duration_ms=duration_ms,
                )

        # 5. Run tool with timeout
        try:
            raw_output = await asyncio.wait_for(
                tool.run(**validated_args),
                timeout=self.tool_timeout_seconds,
            )
        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._audit(tool_name, tier.value, "timeout", duration_ms)
            return ToolExecutionResult(
                success=False,
                tool_name=tool_name,
                error=f"Execution of tool '{tool_name}' timed out after {self.tool_timeout_seconds} seconds.",
                tier=tier.value,
                decision="timeout",
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._audit(tool_name, tier.value, "execution_error", duration_ms)
            return ToolExecutionResult(
                success=False,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' raised an error: {e}",
                tier=tier.value,
                decision="failed",
                duration_ms=duration_ms,
            )

        duration_ms = (time.perf_counter() - start_time) * 1000

        # 6. Apply Secret Filter on Tool Output
        sanitized_output = self._sanitize_output(raw_output)

        # 7. Apply Output Size Limit
        final_output = self._truncate_output(sanitized_output)

        self._audit(
            tool_name=tool_name,
            tier=tier.value,
            decision="executed",
            duration_ms=duration_ms,
            args=validated_args if self.log_content else None,
            result=str(final_output) if self.log_content else None,
        )

        return ToolExecutionResult(
            success=True,
            tool_name=tool_name,
            output=final_output,
            tier=tier.value,
            decision="executed",
            duration_ms=duration_ms,
        )

    def _generate_summary(self, tool: BaseTool, args: dict[str, Any], reason: str) -> str:
        """Generate clear human-readable summary of the action and arguments."""
        arg_items = ", ".join(f"{k}={v!r}" for k, v in args.items())
        return f"{tool.description} with arguments: ({arg_items}) - {reason}"

    def _sanitize_output(self, output: Any) -> Any:
        """Scan tool output against SECRET_PATTERNS and redact if found."""
        if isinstance(output, str):
            sanitized = output
            for pattern in SECRET_PATTERNS:
                sanitized = pattern.sub("[REDACTED_SECRET]", sanitized)
            return sanitized
        elif isinstance(output, dict):
            return {k: self._sanitize_output(v) for k, v in output.items()}
        elif isinstance(output, list):
            return [self._sanitize_output(item) for item in output]
        return output

    def _truncate_output(self, output: Any) -> Any:
        """Truncate string output exceeding maximum allowed characters."""
        if isinstance(output, str) and len(output) > self.output_max_chars:
            return output[: self.output_max_chars] + f"\n... [Truncated: exceeded {self.output_max_chars} chars]"
        return output

    def _audit(
        self,
        tool_name: str,
        tier: str,
        decision: str,
        duration_ms: float,
        args: Any = None,
        result: Any = None,
    ) -> None:
        """Record structured tool execution event to audit log."""
        extra = {
            "tool_name": tool_name,
            "tier": tier,
            "decision": decision,
        }
        if self.log_content:
            extra["arguments"] = args
            extra["result"] = result

        # Log through central audit logger
        self.audit_logger.log(
            endpoint=f"/tool/{tool_name}",
            method="EXEC",
            status_code=200 if decision == "executed" else 403,
            duration_ms=duration_ms,
            extra=extra,
        )
