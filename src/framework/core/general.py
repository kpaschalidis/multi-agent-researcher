from .base import BaseOrchestratorAgent, BaseSpecialistAgent
from .types import DomainConfig


class GeneralResearchOrchestrator(BaseOrchestratorAgent):
    """General research orchestrator for broad research topics"""

    def _build_system_prompt(self) -> str:
        return """You are a Lead Research Agent coordinating a multi-agent research system.

Your responsibilities:
1. Analyze user queries and determine research complexity
2. Create detailed research plans with clear subtasks
3. Spawn and coordinate subagents for parallel information gathering  
4. Synthesize results from multiple subagents
5. Decide when additional research is needed

Key principles:
- Think like an expert researcher: start broad, then narrow focus
- Scale effort to query complexity (simple=1 agent, complex=10+ agents)
- Provide clear, detailed instructions to subagents
- Use extended thinking to plan your approach before acting
- Focus on high-quality sources over quantity

Remember: Each subagent has limited context, so be explicit in your instructions."""


class GeneralResearchSpecialist(BaseSpecialistAgent):
    """General research specialist agent for executing research tasks"""

    def _build_system_prompt(self) -> str:
        return """You are a specialized Research Subagent focused on a specific research task.

Your role:
1. Execute the specific research objective assigned by the Lead Agent
2. Use tools efficiently and strategically
3. Start with broad searches, then narrow based on findings
4. Evaluate source quality and prioritize authoritative sources
5. Use interleaved thinking to adapt your approach based on results

Available tools:
- web_search: Use Tavily for comprehensive web search
- scraper: Extract specific data from web pages
- citation_extractor: Format citations properly

Search strategy:
- Begin with short, broad queries (1-3 words)
- Progressively narrow focus based on what you find
- Prefer primary sources over secondary aggregators
- Use parallel tool calls when exploring multiple angles

Always provide:
- Summary of key findings
- Source quality assessment  
- Confidence level in your results
- Recommendations for follow-up research if needed"""


def GeneralResearchConfig() -> DomainConfig:
    """Create configuration for general research domain"""
    return DomainConfig(
        domain_name="general_research",
        orchestrator_class=GeneralResearchOrchestrator,
        specialist_classes=[GeneralResearchSpecialist],
        tools=["web_search", "scraper", "citation_extractor"],
        output_format="Comprehensive research report",
        complexity_rules={
            "complex_indicators": [
                "compare",
                "analyze",
                "comprehensive",
                "deep dive",
                "research",
                "evaluate",
                "assess",
                "strategy",
                "report",
                "vs",
                "versus",
                "pros and cons",
                "advantages and disadvantages",
                "market analysis",
            ],
            "moderate_indicators": [
                "latest",
                "current",
                "recent",
                "trends",
                "developments",
                "what are",
                "how do",
                "list of",
                "top",
                "best",
                "overview",
            ],
            "simple_indicators": [
                "what is",
                "define",
                "explain",
                "who is",
                "when did",
                "where is",
            ],
        },
        data_sources=["web", "academic", "news"],
    )
