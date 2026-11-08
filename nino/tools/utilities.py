"""Utility tools: current time inspection and safe AST math calculator."""

import ast
import math
import operator
from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field

from nino.tools.base import BaseTool
from nino.tools.permissions import PermissionTier


class CurrentTimeInput(BaseModel):
    timezone_name: str = Field(default="Asia/Kolkata", description="Timezone name or offset (default 'Asia/Kolkata')")


class CurrentTimeTool(BaseTool):
    name = "get_current_time"
    description = "Get the current accurate system date, day of week, and local time. Read-only."
    args_schema = CurrentTimeInput
    tier = PermissionTier.READ_ONLY

    def __init__(
        self,
        default_timezone: str = "Asia/Kolkata",
        location: str = "Vadakara, Calicut, Kerala, India",
    ):
        self.default_timezone = default_timezone
        self.location = location

    async def run(self, timezone_name: str = "Asia/Kolkata") -> str:
        now_local = datetime.now().astimezone()
        now_utc = datetime.now(timezone.utc)

        time_12 = now_local.strftime("%I:%M:%S %p")
        time_24 = now_local.strftime("%H:%M:%S")
        date_str = now_local.strftime("%A, %B %d, %Y")
        tz_str = now_local.strftime("%Z (%z)").strip() or "IST (+0530)"

        return (
            f"Current Local Time: {date_str}, {time_12} ({tz_str})\n"
            f"24-Hour Format: {time_24}\n"
            f"Location: {self.location}\n"
            f"Timezone: {timezone_name or self.default_timezone} ({tz_str})\n"
            f"Current UTC Time: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )


class CalculatorInput(BaseModel):
    expression: str = Field(..., description="Mathematical expression to evaluate (e.g. '((3 + 5) * 2) ** 2')")


# Supported operators for safe AST evaluation
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Supported math functions
SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval_node(node: ast.AST) -> Any:
    """Recursively evaluate an AST node, allowing only safe arithmetic operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    if isinstance(node, ast.BinOp):
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        # Prevent massive exponents causing CPU hangs
        if op_type is ast.Pow and (right > 1000 or (isinstance(left, (int, float)) and abs(left) > 1000 and right > 10)):
            raise ValueError("Exponent too large to evaluate safely.")
        return SAFE_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval_node(node.operand)
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return SAFE_OPERATORS[op_type](operand)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct named mathematical function calls are supported.")
        func_name = node.func.id
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(f"Function '{func_name}' is not allowed in safe calculator.")
        args = [_safe_eval_node(arg) for arg in node.args]
        return SAFE_FUNCTIONS[func_name](*args)

    if isinstance(node, ast.Name):
        if node.id in SAFE_FUNCTIONS and isinstance(SAFE_FUNCTIONS[node.id], (int, float)):
            return SAFE_FUNCTIONS[node.id]
        raise ValueError(f"Unknown variable or identifier: {node.id}")

    raise ValueError(f"Unsupported syntax in math expression: {type(node).__name__}")


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Safely evaluate a mathematical expression. Read-only."
    args_schema = CalculatorInput
    tier = PermissionTier.READ_ONLY

    async def run(self, expression: str) -> str:
        clean_expr = expression.strip()
        if not clean_expr:
            return "Error: Expression cannot be empty."

        try:
            parsed = ast.parse(clean_expr, mode="eval")
            result = _safe_eval_node(parsed)
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {e}"
