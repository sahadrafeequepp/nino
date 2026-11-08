"""Base class and common models for Nino tools."""

from abc import ABC, abstractmethod
from typing import Any, Type
from pydantic import BaseModel

from nino.tools.permissions import PermissionTier


class EmptyToolArgs(BaseModel):
    """Default empty schema for tools with no arguments."""
    pass


class BaseTool(ABC):
    """Abstract base definition for all tools within the Nino execution framework."""

    name: str
    description: str
    args_schema: Type[BaseModel] = EmptyToolArgs
    tier: PermissionTier

    @abstractmethod
    async def run(self, **kwargs: Any) -> Any:
        """Execute the tool logic with validated keyword arguments."""
        pass

    def to_ollama_tool(self) -> dict[str, Any]:
        """Convert tool metadata to the standard Ollama / OpenAI tool schema."""
        schema = self.args_schema.model_json_schema()
        # Clean up schema titles for cleaner LLM prompting
        parameters = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"[{self.tier.value}] {self.description}",
                "parameters": parameters,
            },
        }

    def format_compact_spec(self) -> str:
        """Format a concise, single-line signature and description for efficient prompting."""
        schema = self.args_schema.model_json_schema()
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        params = []
        for p_name, p_info in properties.items():
            p_type = p_info.get("type", "any")
            opt = "" if p_name in required else "?"
            params.append(f"{p_name}{opt}: {p_type}")
        params_str = ", ".join(params)
        short_desc = self.description.split(".")[0].strip()
        return f"- `{self.name}`({params_str}) [{self.tier.value}]: {short_desc}"

    def format_text_spec(self) -> str:
        """Format human and LLM-readable text spec for system prompts."""
        schema = self.args_schema.model_json_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        prop_strs = []
        for prop_name, prop_val in properties.items():
            req_marker = " (required)" if prop_name in required else " (optional)"
            prop_strs.append(f"    - {prop_name}: {prop_val.get('type', 'any')}{req_marker}: {prop_val.get('description', '')}")
        props_formatted = "\n".join(prop_strs) if prop_strs else "    None"
        return f"- Tool: `{self.name}` [{self.tier.value}]\n  Description: {self.description}\n  Parameters:\n{props_formatted}"

