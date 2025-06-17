from typing import List, Dict, Any, TypedDict
from dataclasses import dataclass, field
from enum import Enum
from .logging import AgentLogger, ReasoningChain

from .prompts import (
    RESEARCH_AGENT_PROMPT,
    RESEARCH_LEAD_PROMPT,
    BROWSER_AGENT_PROMPT,
    EDITOR_AGENT_PROMPT,
    REVIEWER_AGENT_PROMPT,
    REVISER_AGENT_PROMPT,
    WRITER_AGENT_PROMPT,
    PUBLISHER_AGENT_PROMPT,
    HIGH_QUALITY_RESEARCH_CRITERIA,
)


@dataclass
class WorkflowPrompts:
    """Prompts for all workflow agents"""

    browser: str = BROWSER_AGENT_PROMPT
    editor: str = EDITOR_AGENT_PROMPT
    research_lead: str = RESEARCH_LEAD_PROMPT
    research_agent: str = RESEARCH_AGENT_PROMPT
    reviewer: str = REVIEWER_AGENT_PROMPT
    reviser: str = REVISER_AGENT_PROMPT
    writer: str = WRITER_AGENT_PROMPT
    publisher: str = PUBLISHER_AGENT_PROMPT
    quality_criteria: str = HIGH_QUALITY_RESEARCH_CRITERIA


@dataclass
class ResearchConfig:
    """Complete configuration for a ResearchWorkflow execution"""

    domain_name: str
    complexity_rules: Dict[str, Any]
    data_sources: List[str]

    workflow_prompts: WorkflowPrompts = field(default_factory=WorkflowPrompts)
    research_focus: str = "comprehensive research and analysis"
    tools: List[str] = field(
        default_factory=lambda: ["web_search", "scraper", "citation_extractor"]
    )
    output_format: str = "Comprehensive Research Report"
    target_audience: str = "Professional stakeholders"

    quality_threshold: float = 0.8
    max_revisions: int = 2

    max_sources_per_task: int = 10
    research_depth: str = (
        "comprehensive"  # "quick", "standard", "comprehensive", "deep"
    )


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
