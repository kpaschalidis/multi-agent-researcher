"""
Multi-Agent Research System using LangGraph/LangChain
Based on Anthropic's multi-agent research architecture
"""

import os
import asyncio
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, TypedDict
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from scrapegraphai.graphs import SmartScraperGraph

from langgraph.graph import StateGraph, END

load_dotenv()


# State Management
class ResearchState(TypedDict):
    query: str
    research_plan: str
    subagent_tasks: List[Dict[str, Any]]
    subagent_results: List[Dict[str, Any]]
    current_step: str
    iteration_count: int
    final_report: str
    citations: List[Dict[str, str]]
    errors: List[str]


class TaskComplexity(Enum):
    SIMPLE = "simple"  # 1 agent, 3-10 tool calls
    MODERATE = "moderate"  # 2-4 agents, 10-15 calls each
    COMPLEX = "complex"  # 10+ agents, specialized tasks


@dataclass
class SubagentTask:
    id: str
    objective: str
    search_queries: List[str]
    tools_to_use: List[str]
    output_format: str
    boundaries: str
    priority: int = 1
    status: str = "pending"
    results: Dict[str, Any] = field(default_factory=dict)


# Tools
class TavilyWebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = """Search the web for comprehensive, recent information. 
    Tavily provides AI-optimized search results with better source quality and relevance.
    Use broad queries first, then narrow down based on findings."""

    search: TavilySearch

    def __init__(self, api_key: str):
        search_instance = TavilySearch(
            max_results=5,
            topic="general",
            api_key=api_key,
        )

        super().__init__(search=search_instance)

    def _run(self, query: str) -> str:
        try:
            results = self.search.invoke(query)

            # Format results for better agent consumption
            formatted_results = []
            for result in results:
                formatted_result = f"""
Source: {result.get('url', 'Unknown')}
Title: {result.get('title', 'No title')}
Content: {result.get('content', 'No content')}
Relevance Score: {result.get('score', 'N/A')}
---"""
                formatted_results.append(formatted_result)

            return f"Search results for '{query}':\n" + "\n".join(formatted_results)
        except Exception as e:
            return f"Search failed for '{query}': {str(e)}"


class ScrapeGraphTool(BaseTool):
    name: str = "intelligent_scraper"
    description: str = """Extract specific information from web pages using AI-powered scraping.
    Provide a URL and describe what information you want to extract.
    Much more effective than basic web scraping for getting structured data."""

    llm_config: Dict[str, Any]

    def __init__(self, llm_config: Dict[str, Any]):
        super().__init__(llm_config=llm_config)

    def _run(self, url: str, extraction_prompt: str) -> str:
        try:
            graph_config = {
                "llm": self.llm_config,
                "verbose": False,
                "headless": True,
            }

            smart_scraper = SmartScraperGraph(
                prompt=extraction_prompt, source=url, config=graph_config
            )

            result = smart_scraper.run()

            return f"Extracted from {url}:\n{json.dumps(result, indent=2)}"

        except Exception as e:
            return f"Scraping failed for {url}: {str(e)}"


class CitationTool(BaseTool):
    name: str = "citation_extractor"
    description: str = "Extract and format citations from research results"

    def _run(self, text: str, sources: List[str]) -> Dict[str, Any]:
        # TODO: Simple citation extraction - in practice, this would be more sophisticated
        citations = []
        for i, source in enumerate(sources, 1):
            citations.append(
                {
                    "id": f"cite_{i}",
                    "source": source,
                    "relevance_score": 0.8,  # TODO: Would be calculated based on content analysis
                }
            )

        return {
            "citations": citations,
            "formatted_text": text,  # TODO: Would include citation markers
        }


# Agent Classes
class LeadResearcher:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.system_prompt = """You are a Lead Research Agent coordinating a multi-agent research system.

Your responsibilities:
1. Analyze user queries and determine research complexity
2. Create detailed research plans with clear subtasks
3. Spawn and coordinate subagents for parallel information gathering  
4. Synthesize results from multiple subagents
5. Decide when additional research is needed

Key principles:
- Think like an expert researcher: start broad, then narrow focus
- Scale effort to query complexity (simple=1 agent, complex=6+ agents)
- Provide clear, detailed instructions to subagents including:
  * Specific objective
  * Output format requirements
  * Tool usage guidance
  * Task boundaries to avoid overlap
- Use extended thinking to plan your approach before acting
- Focus on high-quality sources over quantity

Remember: Each subagent has limited context, so be explicit in your instructions."""

    def determine_complexity(self, query: str) -> TaskComplexity:
        """Determine query complexity using heuristics"""
        query_lower = query.lower()

        complex_indicators = [
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
        ]

        moderate_indicators = [
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
        ]

        simple_indicators = [
            "what is",
            "define",
            "explain",
            "who is",
            "when did",
            "where is",
        ]

        if any(indicator in query_lower for indicator in complex_indicators):
            return TaskComplexity.COMPLEX
        elif any(indicator in query_lower for indicator in moderate_indicators):
            return TaskComplexity.MODERATE
        elif any(indicator in query_lower for indicator in simple_indicators):
            return TaskComplexity.SIMPLE
        else:
            return TaskComplexity.MODERATE

    def get_agent_config(self, complexity: TaskComplexity) -> Dict[str, int]:
        """Get agent configuration based on complexity"""
        configs = {
            TaskComplexity.SIMPLE: {"num_subagents": 1, "max_tool_calls": 5},
            TaskComplexity.MODERATE: {"num_subagents": 3, "max_tool_calls": 10},
            TaskComplexity.COMPLEX: {"num_subagents": 6, "max_tool_calls": 15},
        }
        return configs[complexity]

    def _create_fallback_plan(
        self, query: str, complexity: TaskComplexity, config: Dict[str, int]
    ) -> Dict[str, Any]:
        """Create fallback plan when JSON parsing fails"""
        subtasks = []

        if complexity == TaskComplexity.SIMPLE:
            subtasks = [
                {
                    "id": "task_1",
                    "objective": f"Research and explain: {query}",
                    "search_queries": [query],
                    "tools": ["web_search"],
                    "output_format": "Clear, concise explanation with key facts",
                    "boundaries": "Focus on factual, authoritative information",
                    "priority": 1,
                }
            ]
        elif complexity == TaskComplexity.MODERATE:
            subtasks = [
                {
                    "id": "task_1",
                    "objective": f"Find recent information about: {query}",
                    "search_queries": [query, f"latest {query}", f"recent {query}"],
                    "tools": ["web_search", "intelligent_scraper"],
                    "output_format": "Comprehensive summary with current developments",
                    "boundaries": "Focus on recent, credible sources",
                    "priority": 1,
                },
                {
                    "id": "task_2",
                    "objective": f"Gather background and context for: {query}",
                    "search_queries": [f"{query} background", f"{query} overview"],
                    "tools": ["web_search"],
                    "output_format": "Background information and context",
                    "boundaries": "Provide foundational understanding",
                    "priority": 2,
                },
                {
                    "id": "task_3",
                    "objective": f"Find expert analysis and opinions on: {query}",
                    "search_queries": [f"{query} analysis", f"{query} expert opinion"],
                    "tools": ["web_search", "intelligent_scraper"],
                    "output_format": "Expert insights and analysis",
                    "boundaries": "Focus on authoritative expert sources",
                    "priority": 2,
                },
            ]
        else:
            base_query_parts = query.split()[:3]  # First few words
            aspects = [
                "overview",
                "recent developments",
                "analysis",
                "trends",
                "impact",
                "future outlook",
            ]
            subtasks = [
                {
                    "id": f"task_{i+1}",
                    "objective": f"Research {aspects[i % len(aspects)]} for: {query}",
                    "search_queries": [
                        f"{' '.join(base_query_parts)} {aspects[i % len(aspects)]}",
                        f"{query} {aspects[i % len(aspects)]}",
                    ],
                    "tools": ["web_search", "intelligent_scraper"],
                    "output_format": f"Detailed analysis of {aspects[i % len(aspects)]}",
                    "boundaries": f"Focus specifically on {aspects[i % len(aspects)]}, avoid overlap with other tasks",
                    "priority": 1 if i < 3 else 2,
                }
                for i in range(config["num_subagents"])
            ]

        return {
            "complexity": complexity.value,
            "reasoning": "Fallback plan created due to JSON parsing failure",
            "num_subagents": config["num_subagents"],
            "research_plan": f"Execute {complexity.value} research plan for: {query}",
            "subtasks": subtasks,
        }

    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity and create research plan"""
        # Determine complexity using heuristics
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
            "tools": ["web_search", "intelligent_scraper"],
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
        )
        response = await self.llm.ainvoke(messages)

        try:
            result = json.loads(response.content)
            # Ensure we stick to the predetermined configuration
            result["complexity"] = complexity.value
            result["num_subagents"] = config["num_subagents"]
            return result
        except json.JSONDecodeError:
            # Fallback with proper complexity-based configuration
            return self._create_fallback_plan(query, complexity, config)

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


class SubagentResearcher:
    def __init__(self, llm: ChatOpenAI, tools: List[BaseTool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.system_prompt = """You are a specialized Research Subagent focused on a specific research task.

Your role:
1. Execute the specific research objective assigned by the Lead Agent
2. Use tools efficiently and strategically
3. Start with broad searches, then narrow based on findings
4. Evaluate source quality and prioritize authoritative sources
5. Use interleaved thinking to adapt your approach based on results

Available tools:
- web_search: Use Tavily for comprehensive web search with AI-optimized results
- intelligent_scraper: Extract specific data from web pages using AI-powered scraping
- citation_extractor: Format citations properly

Search strategy:
- Begin with short, broad queries (1-3 words)
- Use web_search for general information gathering
- Use intelligent_scraper when you need specific data from known URLs
- Progressively narrow focus based on what you find
- Prefer primary sources over secondary aggregators
- Avoid SEO-optimized content farms
- Use parallel tool calls when exploring multiple angles

Always provide:
- Summary of key findings
- Source quality assessment  
- Confidence level in your results
- Recommendations for follow-up research if needed"""

    async def execute_task(self, task: SubagentTask) -> Dict[str, Any]:
        """Execute assigned research task"""
        # Create dynamic prompt based on task
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    """Execute this research task:

Objective: {objective}
Suggested Queries: {queries}
Available Tools: {tools}
Output Format: {output_format}
Task Boundaries: {boundaries}

Think through your approach, then execute the research systematically.
Use multiple search queries and evaluate results before proceeding.
Provide your findings in the requested format.""",
                ),
            ]
        )

        messages = prompt.format_messages(
            objective=task.objective,
            queries=", ".join(task.search_queries),
            tools=", ".join(task.tools_to_use),
            output_format=task.output_format,
            boundaries=task.boundaries,
        )

        # Execute research with enhanced tools
        findings = []
        sources = []

        # Use Tavily for comprehensive web search
        for query in task.search_queries:
            if "web_search" in task.tools_to_use and "web_search" in self.tools:
                result = self.tools["web_search"]._run(query)
                findings.append(result)
                sources.append(f"Tavily search: {query}")

        # If we found URLs in the search results, use intelligent scraping for deeper analysis
        if (
            "intelligent_scraper" in task.tools_to_use
            and "intelligent_scraper" in self.tools
        ):
            # Extract URLs from search results (simplified - would be more sophisticated)
            import re

            urls_found = []
            for finding in findings:
                urls = re.findall(r"Source: (https?://[^\s]+)", finding)
                urls_found.extend(urls[:2])  # Limit to 2 URLs per search

            # Scrape the most relevant URLs for detailed information
            for url in urls_found[:3]:  # Limit total scraping
                scrape_prompt = f"Extract key information relevant to: {task.objective}"
                scrape_result = self.tools["intelligent_scraper"]._run(
                    url, scrape_prompt
                )
                findings.append(scrape_result)
                sources.append(f"Scraped content: {url}")

        # Synthesize findings
        all_findings = "\n\n".join(findings)

        synthesis_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    """Analyze and synthesize these research findings:

Task Objective: {objective}
Output Format Required: {output_format}

Raw Findings:
{findings}

Provide a synthesized response that:
1. Directly addresses the objective
2. Highlights key insights
3. Assesses source quality
4. Notes any gaps or limitations
5. Follows the requested output format""",
                ),
            ]
        )

        synthesis_messages = synthesis_prompt.format_messages(
            objective=task.objective,
            output_format=task.output_format,
            findings=all_findings,
        )

        synthesis_response = await self.llm.ainvoke(synthesis_messages)

        return {
            "task_id": task.id,
            "objective": task.objective,
            "findings": synthesis_response.content,
            "sources": sources,
            "confidence": 0.8,  # TODO: should be calculated based on source quality
            "status": "completed",
        }


# LangGraph Workflow
class MultiAgentResearcher:
    def __init__(self, llm: ChatOpenAI, tavily_api_key: str):
        self.llm = llm
        self.tools = [
            TavilyWebSearchTool(tavily_api_key),
            ScrapeGraphTool(
                {
                    "model": "openai/gpt-4o",
                    "api_key": llm.openai_api_key,  # Reuse OpenAI key
                    "temperature": 0.1,
                }
            ),
            CitationTool(),
        ]

        self.lead_agent = LeadResearcher(llm)
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""

        workflow = StateGraph(ResearchState)

        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("spawn_subagents", self._spawn_subagents_node)
        workflow.add_node("execute_research", self._execute_research_node)
        workflow.add_node("synthesize_results", self._synthesize_results_node)
        workflow.add_node("add_citations", self._add_citations_node)

        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "spawn_subagents")
        workflow.add_edge("spawn_subagents", "execute_research")
        workflow.add_edge("execute_research", "synthesize_results")
        workflow.add_edge("synthesize_results", "add_citations")
        workflow.add_edge("add_citations", END)

        return workflow

    async def _analyze_query_node(self, state: ResearchState) -> ResearchState:
        """Analyze query and create research plan"""
        analysis = await self.lead_agent.analyze_query(state["query"])

        state["research_plan"] = analysis["research_plan"]
        state["subagent_tasks"] = analysis["subtasks"]
        state["current_step"] = "query_analyzed"

        return state

    async def _spawn_subagents_node(self, state: ResearchState) -> ResearchState:
        """Prepare subagent tasks"""
        tasks = []
        for task_data in state["subagent_tasks"]:
            task = SubagentTask(
                id=task_data["id"],
                objective=task_data["objective"],
                search_queries=task_data["search_queries"],
                tools_to_use=task_data["tools"],
                output_format=task_data["output_format"],
                boundaries=task_data["boundaries"],
                priority=task_data.get("priority", 1),
            )
            tasks.append(task)

        state["current_step"] = "subagents_spawned"
        return state

    async def _execute_research_node(self, state: ResearchState) -> ResearchState:
        """Execute parallel research with subagents"""
        results = []

        async def execute_single_task(task_data):
            task = SubagentTask(
                id=task_data["id"],
                objective=task_data["objective"],
                search_queries=task_data["search_queries"],
                tools_to_use=task_data["tools"],
                output_format=task_data["output_format"],
                boundaries=task_data["boundaries"],
            )

            subagent = SubagentResearcher(self.llm, self.tools)
            return await subagent.execute_task(task)

        tasks = [execute_single_task(task) for task in state["subagent_tasks"]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                state["errors"].append(str(result))
            else:
                valid_results.append(result)

        state["subagent_results"] = valid_results
        state["current_step"] = "research_completed"

        return state

    async def _synthesize_results_node(self, state: ResearchState) -> ResearchState:
        """Synthesize results into final report"""
        final_report = await self.lead_agent.synthesize_results(
            state["query"], state["subagent_results"]
        )

        state["final_report"] = final_report
        state["current_step"] = "results_synthesized"

        return state

    async def _add_citations_node(self, state: ResearchState) -> ResearchState:
        """Add citations to the final report"""
        all_sources = []
        for result in state["subagent_results"]:
            all_sources.extend(result.get("sources", []))

        citation_tool = CitationTool()
        citation_result = citation_tool._run(state["final_report"], all_sources)

        state["citations"] = citation_result["citations"]
        state["final_report"] = citation_result["formatted_text"]
        state["current_step"] = "completed"

        return state

    async def research(self, query: str) -> Dict[str, Any]:
        """Execute the complete research workflow"""
        initial_state = ResearchState(
            query=query,
            research_plan="",
            subagent_tasks=[],
            subagent_results=[],
            current_step="initialized",
            iteration_count=0,
            final_report="",
            citations=[],
            errors=[],
        )

        app = self.workflow.compile()

        result = await app.ainvoke(initial_state)

        return {
            "query": result["query"],
            "research_plan": result["research_plan"],
            "final_report": result["final_report"],
            "citations": result["citations"],
            "num_subagents": len(result["subagent_results"]),
            "errors": result["errors"],
            "execution_time": datetime.now().isoformat(),
        }


# Usage Example
async def main():
    """Example usage of the multi-agent research system with enhanced tools"""

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        timeout=30,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        print(
            "Warning: TAVILY_API_KEY not found. Please set this environment variable."
        )
        return

    researcher = MultiAgentResearcher(llm, tavily_api_key)

    # Example research queries with different complexities
    queries = [
        # "What is machine learning?",  # Simple
        # "What are the latest developments in AI agent frameworks in 2024?",  # Moderate
        "Compare the market performance of major cloud providers and analyze their competitive advantages",  # Complex
    ]

    for query in queries:
        print(f"\n{'='*50}")
        print(f"Research Query: {query}")
        print(f"{'='*50}")

        try:
            result = await researcher.research(query)

            print(f"\nResearch Plan:")
            print(result["research_plan"])

            print(f"\nFinal Report:")
            print(result["final_report"])

            print(f"\nMetadata:")
            print(f"- Subagents used: {result['num_subagents']}")
            print(f"- Citations: {len(result['citations'])}")
            print(f"- Errors: {len(result['errors'])}")

            if result["errors"]:
                print("- Error details:", result["errors"])

        except Exception as e:
            print(f"Research failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
