from typing import List, Dict, Any, TypedDict
from dataclasses import dataclass, field
from enum import Enum
from .logging import AgentLogger, ReasoningChain


class TaskComplexity(Enum):
    """Defines the complexity levels for research tasks"""

    SIMPLE = "simple"  # example: 1 agent, 3-10 tool calls
    MODERATE = "moderate"  # example: 2-4 agents, 10-15 calls each
    COMPLEX = "complex"  # example: 10+ agents, specialized tasks


class ResearchState(TypedDict):
    """State management for the research workflow"""

    query: str
    research_plan: str
    subagent_tasks: List[Dict[str, Any]]
    subagent_results: List[Dict[str, Any]]
    current_step: str
    iteration_count: int
    final_report: str
    citations: List[Dict[str, str]]
    errors: List[str]
    logger: "AgentLogger"
    reasoning_history: List["ReasoningChain"]
    initial_research: str = ""
    research_outline: Dict[str, Any] = None
    quality_guidelines: List[str] = None
    review_feedback: List[str] = None
    revision_history: List[Dict] = None
    quality_score: float = 0.0
    needs_revision: bool = False
    publication_formats: List[str] = None

    def __post_init__(self):
        if self.review_feedback is None:
            self.review_feedback = []
        if self.revision_history is None:
            self.revision_history = []
        if self.publication_formats is None:
            self.publication_formats = ["markdown"]


@dataclass
class SubagentTask:
    """Represents a task assigned to a specialist agent"""

    id: str
    objective: str
    search_queries: List[str]
    tools_to_use: List[str]
    output_format: str
    boundaries: str
    priority: int = 1
    status: str = "pending"
    results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainConfig:
    """Configuration for domain-specific research"""

    domain_name: str
    research_lead_class: type
    research_agent_classes: List[type]
    tools: List[str]
    output_format: str
    complexity_rules: Dict[str, Any]
    data_sources: List[str] = field(default_factory=lambda: ["web"])

    def validate(self) -> bool:
        """Validate the domain configuration"""
        required_fields = [
            "domain_name",
            "research_lead_class",
            "research_lead_class",
            "tools",
            "output_format",
            "complexity_rules",
        ]

        for field_name in required_fields:
            if not getattr(self, field_name):
                raise ValueError(f"Missing required field: {field_name}")

        if not self.research_agent_classes:
            raise ValueError("At least one specialist class must be provided")

        if not self.tools:
            raise ValueError("At least one tool must be specified")

        return True
