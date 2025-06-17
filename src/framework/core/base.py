import json
import json5
import re
from typing import List, Dict, Any
from datetime import datetime

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .types import ResearchConfig, SubagentTask, TaskComplexity
from .logging import AgentLogger, LogLevel, ReasoningChain
from .constants import COMPLEXITY_INDICATORS, ALL_COMPLEXITY_VALUES, AGENT_CONFIGS


class ResearchLead:
    """Base class for orchestrator agents that manage research workflows"""

    def __init__(
        self,
        llm: ChatOpenAI,
        research_config: ResearchConfig,
        agent_id: str = None,
        logger: AgentLogger = None,
    ):
        self.llm = llm
        self.research_config = research_config
        self.logger = logger or AgentLogger()
        self.agent_id = agent_id or f"research_lead_{id(self)}"
        self.system_prompt = self.research_config.workflow_prompts.research_lead

    def determine_complexity(self, query: str) -> TaskComplexity:
        """Determine query complexity using domain-specific heuristics"""

        self.logger.log(
            LogLevel.AGENT,
            self.agent_id,
            "complexity_analysis",
            f"Analyzing query complexity for: '{query[:50]}...'",
        )

        query_lower = query.lower()
        rules = self.research_config.complexity_rules

        complex_indicators = rules.get(
            "complex_indicators", COMPLEXITY_INDICATORS["complex"]
        )
        moderate_indicators = rules.get(
            "moderate_indicators", COMPLEXITY_INDICATORS["moderate"]
        )
        simple_indicators = rules.get(
            "simple_indicators", COMPLEXITY_INDICATORS["simple"]
        )

        matched_indicators = []
        complexity = TaskComplexity.MODERATE

        if any(indicator in query_lower for indicator in complex_indicators):
            matched_indicators = [
                ind for ind in complex_indicators if ind in query_lower
            ]
            complexity = TaskComplexity.COMPLEX
        elif any(indicator in query_lower for indicator in moderate_indicators):
            matched_indicators = [
                ind for ind in moderate_indicators if ind in query_lower
            ]
            complexity = TaskComplexity.MODERATE
        elif any(indicator in query_lower for indicator in simple_indicators):
            matched_indicators = [
                ind for ind in simple_indicators if ind in query_lower
            ]
            complexity = TaskComplexity.SIMPLE

        # Logs
        reasoning_chain = ReasoningChain(
            agent_id=self.agent_id,
            task_id="complexity_determination",
            timestamp=datetime.now().isoformat(),
            thinking_process=[
                f"Analyzing query: '{query}'",
                f"Checking for complexity indicators in: {query_lower}",
                f"Found indicators: {matched_indicators}",
                f"Matched pattern suggests: {complexity.value}",
            ],
            decision_rationale=f"Query contains {matched_indicators} which indicates {complexity.value} research task",
            selected_action=f"Set complexity to {complexity.value}",
            alternatives_considered=[
                c for c in ALL_COMPLEXITY_VALUES if c != complexity.value
            ],
            confidence_level=0.8 if matched_indicators else 0.6,
            expected_outcome=f"Will spawn {self.get_agent_config(complexity)['num_subagents']} subagents",
        )

        self.logger.log_reasoning(reasoning_chain)

        self.logger.log(
            LogLevel.RESEARCH,
            self.agent_id,
            "complexity_determined",
            f"Query complexity: {complexity.value}",
            data={
                "complexity": complexity.value,
                "matched_indicators": matched_indicators,
                "agent_config": self.get_agent_config(complexity),
            },
        )

        return complexity

    def get_agent_config(self, complexity: TaskComplexity) -> Dict[str, int]:
        """Get agent configuration based on complexity"""
        return AGENT_CONFIGS[complexity]

    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity and create research plan"""
        complexity = self.determine_complexity(query)
        config = self.get_agent_config(complexity)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    """Analyze this research query and create a detailed plan:

Query: {query}
Predetermined Complexity: {complexity}
Number of Subagents: {num_subagents}
Max Tool Calls per Agent: {max_tool_calls}

Create a research plan with exactly {num_subagents} subtasks. Provide your analysis in this JSON format:
{{
    "complexity": "{complexity}",
    "reasoning": "explanation of why this complexity level fits",
    "num_subagents": {num_subagents},
    "research_plan": "detailed multi-step plan",
    "subtasks": [
        {{
            "id": "task_1",
            "objective": "specific goal",
            "search_queries": ["query1", "query2"],
            "tools": {tools},
            "output_format": "format description",
            "boundaries": "what NOT to include",
            "priority": 1
        }}
    ]
}}""",
                ),
            ]
        )

        messages = prompt.format_messages(
            query=query,
            complexity=complexity.value,
            num_subagents=config["num_subagents"],
            max_tool_calls=config["max_tool_calls"],
            tools=self.research_config.tools,
        )

        try:
            response = await self.llm.ainvoke(messages)
            raw = response.content.strip()

            if raw.startswith("```"):
                raw = raw.replace("```json", "").replace("```", "").strip()
            result = json5.loads(raw)

            result["complexity"] = complexity.value
            result["num_subagents"] = config["num_subagents"]

            return result
        except json.JSONDecodeError:
            self.logger.log(
                LogLevel.ERROR,
                self.agent_id,
                "json_decode_error",
                f"❌ JSON parsing failed for query: {query}",
            )
            return self._create_fallback_plan(query, complexity, config)

    def _create_fallback_plan(
        self, query: str, complexity: TaskComplexity, config: Dict[str, int]
    ) -> Dict[str, Any]:
        """Create fallback plan when JSON parsing fails

        This is a fallback approach to create a research plan when the JSON parsing fails.
        It creates a research plan with 6 subtasks, each with a different research aspect.
        The research aspects are:
        - root causes and mechanisms
        - current impacts and evidence
        - future projections
        - solutions and mitigation
        """

        research_aspects = [
            {
                "aspect": "root causes and mechanisms",
                "queries": [
                    f"{query} causes",
                    f"{query} mechanisms",
                    f"how does {query} work",
                ],
            },
            {
                "aspect": "current impacts and evidence",
                "queries": [
                    f"{query} impacts",
                    f"{query} evidence",
                    f"current effects {query}",
                ],
            },
            {
                "aspect": "future projections",
                "queries": [
                    f"{query} future",
                    f"{query} predictions",
                    f"{query} projections",
                ],
            },
            {
                "aspect": "solutions and mitigation",
                "queries": [
                    f"{query} solutions",
                    f"{query} mitigation",
                    f"how to solve {query}",
                ],
            },
            {
                "aspect": "economic implications",
                "queries": [
                    f"{query} economic impact",
                    f"{query} cost",
                    f"{query} economic effects",
                ],
            },
            {
                "aspect": "policy and governance",
                "queries": [
                    f"{query} policy",
                    f"{query} governance",
                    f"{query} regulation",
                ],
            },
        ]

        subtasks = []

        for i, aspect in enumerate(research_aspects):
            subtasks.append(
                {
                    "id": f"task_{i+1}",
                    "objective": f"Research {aspect['aspect']} of: {query}",
                    "search_queries": aspect["queries"][
                        :2
                    ],  # Limit to 2 queries per task
                    "tools": self.research_config.tools,
                    "output_format": f"Comprehensive analysis of {aspect['aspect']}",
                    "boundaries": f"Focus specifically on {aspect['aspect']}, avoid overlap with other research areas",
                    "priority": 1 if i < 3 else 2,  # First 3 tasks high priority
                }
            )

        return {
            "complexity": complexity.value,
            "reasoning": "Fallback plan created due to JSON parsing failure",
            "num_subagents": len(subtasks),
            "research_plan": f"Execute {complexity.value} research plan for: {query}",
            "subtasks": subtasks,
        }

    async def synthesize_results(self, query: str, subagent_results: List[Dict]) -> str:
        """Synthesize results from multiple subagents into final report"""
        results_text = "\n\n".join(
            [
                f"Subagent {r['task_id']} Results:\n{r['findings']}"
                for r in subagent_results
            ]
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    """Synthesize these research results into a comprehensive final report:

Original Query: {query}

Subagent Results:
{results}

Create a well-structured report that:
1. Directly answers the original query
2. Integrates findings from all subagents
3. Identifies any gaps or contradictions
4. Provides actionable insights
5. Notes limitations or areas needing further research

Format as a professional research report with clear sections.""",
                ),
            ]
        )

        messages = prompt.format_messages(query=query, results=results_text)
        response = await self.llm.ainvoke(messages)
        return response.content


class ResearchAgent:
    """Base class for domain-specific research agents"""

    def __init__(
        self,
        llm: ChatOpenAI,
        tools: List[BaseTool],
        research_config: ResearchConfig,
        agent_id: str = None,
        logger: AgentLogger = None,
    ):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.research_config = research_config
        self.agent_id = agent_id or f"research_agent_{id(self)}"
        self.logger = logger or AgentLogger()
        self.system_prompt = self.research_config.workflow_prompts.research_agent

    async def execute_task(self, task: SubagentTask) -> Dict[str, Any]:
        """Execute assigned research task with detailed logging"""

        self.logger.log(
            LogLevel.AGENT,
            self.agent_id,
            "task_started",
            f"Starting research task: {task.objective[:50]}...",
            data={"task_id": task.id, "search_queries": task.search_queries},
        )

        reasoning_chain = ReasoningChain(
            agent_id=self.agent_id,
            task_id=task.id,
            timestamp=datetime.now().isoformat(),
            thinking_process=[
                f"Received task: {task.objective}",
                f"Available tools: {task.tools_to_use}",
                f"Need to search for: {task.search_queries}",
                "Planning systematic approach: search -> scrape -> synthesize",
            ],
            decision_rationale="Will use web search first to gather broad information, then scrape specific URLs for details",
            selected_action="Execute multi-step research with tool chaining",
            alternatives_considered=["Direct synthesis", "Single tool approach"],
            confidence_level=0.9,
            expected_outcome="Comprehensive findings addressing the objective",
        )
        self.logger.log_reasoning(reasoning_chain)

        findings = []
        sources = []

        # Web search
        for query in task.search_queries:
            if "web_search" in task.tools_to_use and "web_search" in self.tools:
                self.logger.log(
                    LogLevel.TOOL,
                    self.agent_id,
                    "tool_call",
                    f"Executing web search: '{query}'",
                    data={"tool": "web_search", "query": query},
                )

                result = self.tools["web_search"]._run(query)
                findings.append(result)
                sources.append(f"Web search: {query}")

                self.logger.log(
                    LogLevel.TOOL,
                    self.agent_id,
                    "tool_result",
                    f"Web search completed for: '{query}'",
                    data={
                        "result_length": len(result),
                        "sources_found": result.count("Source:"),
                    },
                )

        # Scraping
        if "scraper" in task.tools_to_use and "scraper" in self.tools:

            urls_found = []
            for finding in findings:
                urls = re.findall(r"Source: (https?://[^\s]+)", finding)
                urls_found.extend(urls[:5])

            self.logger.log(
                LogLevel.AGENT,
                self.agent_id,
                "scraping_planning",
                f"Found {len(urls_found)} URLs to scrape",
                data={"urls": urls_found[:5]},
            )

            for url in urls_found[:5]:
                scrape_prompt = f"""Extract comprehensive information relevant to: {task.objective}

                    Provide detailed, substantive content with specific facts and examples.
                    """

                self.logger.log(
                    LogLevel.TOOL,
                    self.agent_id,
                    "tool_call",
                    f"Scraping URL: {url[:50]}...",
                    data={"tool": "scraper", "url": url},
                )

                scrape_result = await self.tools["scraper"]._arun(url, scrape_prompt)
                findings.append(scrape_result)
                sources.append(f"Scraped content: {url}")

                self.logger.log(
                    LogLevel.TOOL,
                    self.agent_id,
                    "tool_result",
                    f"Scraping completed for: {url[:50]}...",
                    data={"result_length": len(scrape_result)},
                )

        # Synthesize findings
        self.logger.log(
            LogLevel.AGENT,
            self.agent_id,
            "synthesis_start",
            f"Synthesizing {len(findings)} research findings",
            data={"findings_count": len(findings), "sources_count": len(sources)},
        )

        all_findings = "\n\n".join(findings)

        synthesis_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    """Conduct a comprehensive analysis of these research findings.

Task Objective: {objective}
Output Format Required: {output_format}

Research Findings:
{findings}

Create a detailed, comprehensive analysis that includes:

Provide a synthesized response that:
1. Directly addresses the objective
2. Highlights key insights
3. Assesses source quality
4. Notes any gaps or limitations

REQUIREMENTS:
- Include specific examples and case studies
- Cite quantitative data when available
- Use clear subheadings for organization
- Maintain academic rigor and depth
- Ensure all claims are supported by evidence from the research

Focus on providing substantial, detailed content that demonstrates deep understanding of the topic.""",
                ),
            ]
        )

        synthesis_messages = synthesis_prompt.format_messages(
            objective=task.objective,
            output_format=task.output_format,
            findings=all_findings,
        )

        synthesis_response = await self.llm.ainvoke(
            synthesis_messages,
            max_tokens=4000,
            temperature=0.3,
        )

        confidence = 0.8  # TODO: Should be calculated based on source quality

        self.logger.log(
            LogLevel.AGENT,
            self.agent_id,
            "task_completed",
            f"Research task completed: {task.id}",
            data={
                "findings_length": len(synthesis_response.content),
                "sources_used": len(sources),
                "confidence": confidence,
            },
            confidence=confidence,
        )

        return {
            "task_id": task.id,
            "objective": task.objective,
            "findings": synthesis_response.content,
            "sources": sources,
            "confidence": confidence,
            "status": "completed",
        }


def GeneralResearchConfig(domain_name: str = "general_research") -> ResearchConfig:
    """Create configuration for general research domain"""
    return ResearchConfig(
        domain_name=domain_name,
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
