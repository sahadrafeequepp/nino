"""Multi-turn function calling orchestrator for Nino."""

import json
import re
import time
from typing import Any
from pydantic import BaseModel, Field

from nino.models.base import LLMAdapter
from nino.tools.confirmation import Confirmation
from nino.tools.executor import CentralExecutor, ToolExecutionResult
from nino.tools.registry import ToolRegistry


class ToolCallRecord(BaseModel):
    """Record of a tool call executed during the orchestration loop."""

    tool_name: str
    arguments: dict[str, Any]
    decision: str
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0


class OrchestratorResult(BaseModel):
    """Result of the multi-turn tool calling orchestration loop."""

    reply: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    pending_confirmation: Confirmation | None = None
    iterations: int = 0
    total_duration_ms: float = 0.0


# Regex patterns to detect JSON tool calls in fallback text responses
JSON_BLOCK_REGEX = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
RAW_TOOL_JSON_REGEX = re.compile(r'\{\s*"tool"\s*:\s*".*?"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}', re.DOTALL)


class ToolOrchestrator:
    """Orchestrates model tool-calling loops with timeouts, retry, and confirmation handling."""

    def __init__(
        self,
        registry: ToolRegistry,
        executor: CentralExecutor,
        max_iterations: int = 5,
        total_timeout: float = 180.0,
    ):
        self.registry = registry
        self.executor = executor
        self.max_iterations = max_iterations
        self.total_timeout = total_timeout
        self._unsupported_tool_models: set[str] = {"gemma2", "gemma2:2b", "gemma"}

    def build_tool_system_prompt(self, user_message: str | None = None) -> str:
        """Construct prompt instructions informing LLM of available tools and call format."""
        specs = self.registry.get_system_prompt_spec(query=user_message, compact=True)
        return (
            "You have access to the following tools to help answer questions and take actions:\n\n"
            f"{specs}\n\n"
            "INSTRUCTIONS FOR CALLING TOOLS:\n"
            "- If you need to use a tool, respond with ONLY a JSON block in this exact format:\n"
            "```json\n"
            '{\n  "tool": "tool_name",\n  "arguments": {\n    "arg_name": "arg_value"\n  }\n}\n'
            "```\n"
            "- If you do not need to call any tools, or after receiving all necessary tool results, "
            "provide your final helpful answer directly in natural text without unnecessary follow-up questions."
        )


    async def run(
        self,
        messages: list[dict[str, Any]],
        adapter: LLMAdapter,
        confirmation_id: str | None = None,
        context_data: Any = None,
        model_override: str | None = None,
    ) -> OrchestratorResult:
        """Run the multi-turn function calling loop."""
        start_time = time.perf_counter()
        tool_records: list[ToolCallRecord] = []
        current_messages = list(messages)
        iterations = 0
        has_retried_malformed = False

        while iterations < self.max_iterations:
            iterations += 1

            # Check total timeout
            elapsed = time.perf_counter() - start_time
            if elapsed >= self.total_timeout:
                return OrchestratorResult(
                    reply=f"Tool execution reached the maximum allowed time ({self.total_timeout}s).",
                    tool_calls=tool_records,
                    iterations=iterations,
                    total_duration_ms=elapsed * 1000,
                )

            tool_name: str | None = None
            tool_args: dict[str, Any] | None = None
            parse_error: str | None = None
            content: str = ""

            # Check if direct tool call in user message on turn 1
            if iterations == 1 and current_messages and current_messages[-1].get("role") == "user":
                user_direct_call = self._extract_tool_json(current_messages[-1].get("content", ""))
                if user_direct_call and "tool" in user_direct_call:
                    tool_name = user_direct_call.get("tool")
                    raw_args = user_direct_call.get("arguments")
                    if isinstance(raw_args, dict):
                        tool_args = raw_args
                    elif raw_args is None:
                        tool_args = {}
                    else:
                        parse_error = "Tool arguments must be a JSON object."

            # If not a direct tool call from user, call model with available tools
            if not tool_name and not parse_error:
                ollama_tools = self.registry.to_ollama_tools()
                model_kwargs = {"return_raw_message": True}
                if model_override:
                    model_kwargs["model"] = model_override
                active_model = model_override or getattr(adapter, "model_name", "default")
                model_base = str(active_model).split(":")[0].lower()
                can_use_native_tools = (
                    bool(ollama_tools)
                    and active_model not in getattr(self, "_unsupported_tool_models", set())
                    and model_base not in getattr(self, "_unsupported_tool_models", set())
                )

                if can_use_native_tools:
                    model_kwargs["tools"] = ollama_tools

                try:
                    raw_response = await adapter.chat(current_messages, **model_kwargs)
                except Exception as e:
                    # If native tool calling fails with model error (e.g. model doesn't support tools), fall back
                    if "tools" in model_kwargs:
                        if hasattr(self, "_unsupported_tool_models"):
                            self._unsupported_tool_models.add(active_model)
                            self._unsupported_tool_models.add(model_base)
                        del model_kwargs["tools"]
                        raw_response = await adapter.chat(current_messages, **model_kwargs)
                    else:
                        raise e

                # Extract content and native tool_calls if present
                if isinstance(raw_response, dict):
                    content = raw_response.get("content", "") or ""
                    native_tool_calls = raw_response.get("tool_calls", [])
                else:
                    content = str(raw_response)
                    native_tool_calls = []

                if native_tool_calls and isinstance(native_tool_calls, list) and len(native_tool_calls) > 0:
                    first_call = native_tool_calls[0]
                    fn = first_call.get("function", {})
                    tool_name = fn.get("name")
                    raw_args = fn.get("arguments", {})
                    if isinstance(raw_args, str):
                        try:
                            tool_args = json.loads(raw_args)
                        except Exception as e:
                            parse_error = f"Malformed native arguments JSON: {e}"
                    elif isinstance(raw_args, dict):
                        tool_args = raw_args
                else:
                    # Inspect content for fallback JSON tool call
                    parsed_json = self._extract_tool_json(content)
                    if parsed_json:
                        tool_name = parsed_json.get("tool")
                        raw_args = parsed_json.get("arguments")
                        if isinstance(raw_args, dict):
                            tool_args = raw_args
                        elif raw_args is None:
                            tool_args = {}
                        else:
                            parse_error = "Tool arguments must be a JSON object."
                    elif "```json" in content and '"tool"' in content:
                        parse_error = "Malformed tool JSON block."


            # Sanitize tool name if extracted
            if tool_name and isinstance(tool_name, str):
                tool_name = tool_name.strip(" `\"'")
                if "(" in tool_name:
                    tool_name = tool_name.split("(")[0].strip()
                if tool_name.startswith("tools."):
                    tool_name = tool_name[6:].strip()
                elif tool_name.startswith("functions."):
                    tool_name = tool_name[10:].strip()

            # 3. Handle malformed tool call retry
            if parse_error:
                if not has_retried_malformed:
                    has_retried_malformed = True
                    current_messages.append({"role": "assistant", "content": content})
                    current_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"[SYSTEM NOTICE: Your tool call was malformed: {parse_error}. "
                                'Please respond with ONLY a valid JSON object matching: '
                                '{"tool": "<name>", "arguments": {<args>}}]'
                            ),
                        }
                    )
                    continue
                else:
                    # Retry failed, fail gracefully
                    elapsed = (time.perf_counter() - start_time) * 1000
                    return OrchestratorResult(
                        reply=f"Unable to process tool call: {parse_error}\n\n{content}",
                        tool_calls=tool_records,
                        iterations=iterations,
                        total_duration_ms=elapsed,
                    )

            # 4. If no tool was requested, the model gave its final natural reply
            if not tool_name:
                elapsed = (time.perf_counter() - start_time) * 1000
                return OrchestratorResult(
                    reply=content,
                    tool_calls=tool_records,
                    iterations=iterations,
                    total_duration_ms=elapsed,
                )

            # 5. Execute the requested tool via CentralExecutor
            tool_args = tool_args or {}
            exec_result: ToolExecutionResult = await self.executor.execute(
                tool_name=tool_name,
                arguments=tool_args,
                context=current_messages,
                confirmation_id=confirmation_id,
            )

            record = ToolCallRecord(
                tool_name=tool_name,
                arguments=tool_args,
                decision=exec_result.decision,
                output=exec_result.output,
                error=exec_result.error,
                duration_ms=exec_result.duration_ms,
            )
            tool_records.append(record)

            # 6. Check if interactive confirmation is required
            if exec_result.requires_confirmation and exec_result.confirmation:
                elapsed = (time.perf_counter() - start_time) * 1000
                conf = exec_result.confirmation
                reply_msg = (
                    f"Action requires your confirmation:\n\n"
                    f"**Tool**: `{conf.tool_name}` [{conf.tier}]\n"
                    f"**Action**: {conf.summary}\n"
                    f"**Confirmation ID**: `{conf.id}`\n\n"
                    "Please approve or deny this action to proceed."
                )
                return OrchestratorResult(
                    reply=reply_msg,
                    tool_calls=tool_records,
                    pending_confirmation=conf,
                    iterations=iterations,
                    total_duration_ms=elapsed,
                )

            # 7. Feed tool result back into conversation history for the next turn
            result_payload = (
                json.dumps(exec_result.output)
                if not isinstance(exec_result.output, str)
                else exec_result.output
            )
            if not exec_result.success:
                result_payload = f"Tool error: {exec_result.error}"

            # Append assistant's tool call and user's tool result to context
            current_messages.append({"role": "assistant", "content": content})
            current_messages.append(
                {
                    "role": "user",
                    "content": f"[Tool Result for '{tool_name}']\n{result_payload}",
                }
            )

            # Reset confirmation_id so subsequent actions don't reuse the single-use token
            confirmation_id = None

        elapsed = (time.perf_counter() - start_time) * 1000
        return OrchestratorResult(
            reply="Tool execution loop completed maximum allowed iterations.",
            tool_calls=tool_records,
            iterations=iterations,
            total_duration_ms=elapsed,
        )

    def _extract_tool_json(self, text: str) -> dict[str, Any] | None:
        """Extract JSON tool call dictionary from text content, handling nested braces correctly."""
        if not text:
            return None

        # 1. If text contains code fences, check inside code blocks first
        if "```" in text:
            parts = text.split("```")
            for i in range(1, len(parts), 2):
                block = parts[i]
                if block.startswith("json"):
                    block = block[4:]
                parsed = self._try_parse_tool_json(block.strip())
                if parsed:
                    return parsed

        # 2. Try parsing text between first '{' and last '}'
        return self._try_parse_tool_json(text)

    def _try_parse_tool_json(self, text: str) -> dict[str, Any] | None:
        """Find substring from first '{' to last '}' and parse as JSON."""
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = text[start_idx : end_idx + 1]
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and "tool" in data:
                    return data
            except json.JSONDecodeError:
                pass
        return None

