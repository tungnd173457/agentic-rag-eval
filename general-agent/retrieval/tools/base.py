"""Tool plumbing for the agent: spec, result, registry.

Tools are plain Python callables returning ToolResult — stateless, idempotent,
testable without any LLM. The registry renders OpenAI tool format (works for
both providers) and guarantees execute() NEVER raises: invalid args, unknown
tools and internal failures all come back as guidance text the model can act
on (tool results are a steering surface).

The validator covers exactly the JSON-Schema subset our tools use (object with
string, array-of-(enum-)string and array-of-integer properties, plus required
and additionalProperties) — deliberately not a full jsonschema dependency.
Types outside this subset (top-level number, nested objects) pass through
UNVALIDATED — extend _validate before adding a tool parameter of a new type.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger(__name__)


@dataclass
class ToolResult:
    """Rendered, token-capped text for the model + structured source refs for
    the orchestrator (sources[] mapping, tool_result SSE summaries)."""

    text: str
    sources: list[dict] = field(default_factory=list)  # {chunk_id, title, domain}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict                      # JSON Schema (OpenAI "parameters")
    execute: Callable[..., ToolResult]


def _validate(args: dict, schema: dict) -> list[str]:
    """Return human-readable problems ([] = valid) for our schema subset."""
    problems: list[str] = []
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            problems.append(f"'{key}' is required")
    if not schema.get("additionalProperties", True):
        for key in args:
            if key not in props:
                problems.append(f"unknown parameter '{key}'")
    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            continue
        kind = spec.get("type")
        if kind == "string":
            if not isinstance(value, str):
                problems.append(f"'{key}' must be a string")
            elif "enum" in spec and value not in spec["enum"]:
                problems.append(f"'{key}' must be one of: {', '.join(spec['enum'])}")
        elif kind == "array":
            if not isinstance(value, list):
                problems.append(f"'{key}' must be an array")
                continue
            item_spec = spec.get("items", {})
            item_kind = item_spec.get("type", "string")
            enum = item_spec.get("enum")
            minimum = item_spec.get("minimum")
            for item in value:
                if item_kind == "integer":
                    # bool is an int subclass — reject it explicitly.
                    if not isinstance(item, int) or isinstance(item, bool):
                        problems.append(f"'{key}' items must be integers")
                    elif minimum is not None and item < minimum:
                        problems.append(f"'{key}' items must be >= {minimum}")
                elif not isinstance(item, str):
                    problems.append(f"'{key}' items must be strings")
                elif enum and item not in enum:
                    problems.append(
                        f"'{key}' contains invalid value '{item}'. "
                        f"Valid values: {', '.join(enum)}"
                    )
        elif kind == "boolean" and not isinstance(value, bool):
            problems.append(f"'{key}' must be a boolean")
    return problems


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec]):
        self._specs: dict[str, ToolSpec] = {s.name: s for s in specs}

    def openai_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.input_schema,
                },
            }
            for s in self._specs.values()
        ]

    def invalid_args_guidance(self, name: str) -> str:
        """Guidance for arguments that did not even parse as JSON."""
        spec = self._specs.get(name)
        if spec is None:
            return self._unknown_tool_text(name)
        return (
            f"The arguments for {name} were not valid JSON. "
            f"Call it again with arguments matching this schema:\n"
            f"{json.dumps(spec.input_schema, ensure_ascii=False)}"
        )

    def execute(
        self, name: str, args: dict, injected: dict | None = None
    ) -> ToolResult:
        """``args`` are the model-supplied (and validated) arguments; ``injected``
        is orchestrator-owned context (e.g. the seen-chunk set kb_search dedupes
        against) merged in AFTER validation — it is never validated against the
        model-facing schema and the model never sees it."""
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult(text=self._unknown_tool_text(name))
        problems = _validate(args, spec.input_schema)
        if problems:
            return ToolResult(text=(
                f"Invalid arguments for {name}: {'; '.join(problems)}. "
                f"Expected schema:\n{json.dumps(spec.input_schema, ensure_ascii=False)}"
            ))
        try:
            return spec.execute(**args, **(injected or {}))
        except Exception:
            log.exception("tool.execute_failed", tool=name)
            return ToolResult(text=(
                f"The {name} tool failed internally. Do not retry with the same "
                "arguments. Answer from the information you already have, or try "
                "a different tool call, and mention the limitation if relevant."
            ))

    def _unknown_tool_text(self, name: str) -> str:
        return (
            f"Unknown tool '{name}'. Available tools: "
            f"{', '.join(self._specs)}. Call one of those instead."
        )
