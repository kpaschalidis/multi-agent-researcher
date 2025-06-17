from .core.types import TaskComplexity, SubagentTask, DomainConfig, ResearchState
from .core.logging import AgentLogger, LogLevel, LogEntry, ReasoningChain
from .core.constants import COMPLEXITY_INDICATORS, ALL_COMPLEXITY_VALUES, AGENT_CONFIGS
from .core.base import ResearchLead, ResearchAgent
from .tools import TavilyWebSearchTool, CitationTool
from .core.general import (
    GeneralResearchConfig,
    GeneralResearchLead,
    GeneralResearchAgent,
)
from .research_workflow import ResearchWorkflow

__all__ = [
    "TaskComplexity",
    "SubagentTask",
    "DomainConfig",
    "ResearchState",
    "AgentLogger",
    "LogLevel",
    "LogEntry",
    "ReasoningChain",
    "COMPLEXITY_INDICATORS",
    "ALL_COMPLEXITY_VALUES",
    "AGENT_CONFIGS",
    "ResearchWorkflow",
    "ResearchLead",
    "ResearchAgent",
    "GeneralResearchConfig",
    "GeneralResearchLead",
    "GeneralResearchAgent",
    "TavilyWebSearchTool",
    "CitationTool",
    "create_general_research_workflow",
]


def create_general_research_workflow(
    llm, tavily_api_key, verbose_logging=True, log_file=None
):
    """Quick setup function for general research"""
    config = GeneralResearchConfig()
    return ResearchWorkflow(
        llm=llm,
        domain_config=config,
        tavily_api_key=tavily_api_key,
        verbose_logging=verbose_logging,
        log_file=log_file,
    )
