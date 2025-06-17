from .types import TaskComplexity

COMPLEXITY_INDICATORS = {
    "complex": [
        "compare",
        "analyze",
        "comprehensive",
        "deep dive",
        "research",
        "evaluate",
        "assess",
        "strategy",
        "report",
        "investigate",
        "thorough",
        "in-depth",
    ],
    "moderate": [
        "latest",
        "current",
        "recent",
        "trends",
        "developments",
        "what are",
        "how do",
        "list of",
        "overview",
        "summary",
        "update",
    ],
    "simple": [
        "what is",
        "define",
        "explain",
        "who is",
        "when did",
        "where is",
        "how many",
        "basic",
    ],
}

# Pre-computed complexity values for reasoning chains
ALL_COMPLEXITY_VALUES = [c.value for c in TaskComplexity]

# Default agent configurations
AGENT_CONFIGS = {
    TaskComplexity.SIMPLE: {"num_subagents": 1, "max_tool_calls": 5},
    TaskComplexity.MODERATE: {"num_subagents": 3, "max_tool_calls": 10},
    TaskComplexity.COMPLEX: {"num_subagents": 6, "max_tool_calls": 15},
}

# Logging defaults
DEFAULT_LOG_FORMAT = "{timestamp} [{level}] {agent_id}: {message}"
MAX_REASONING_HISTORY = 100
