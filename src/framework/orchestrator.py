import asyncio
from typing import List, Dict, Any
from datetime import datetime

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from .core.logging import AgentLogger, LogLevel
from .core.types import ResearchState, DomainConfig, SubagentTask
from .tools import TavilyWebSearchTool, ScrapeGraphTool, CitationTool


class MultiAgentResearchOrchestrator:
    """Generic framework for any research domain"""

    def __init__(
        self,
        llm: ChatOpenAI,
        domain_config: DomainConfig,
        tavily_api_key: str,
        verbose_logging: bool = True,
        log_file: str = None,
    ):
        self.llm = llm
        self.domain_config = domain_config
        self.domain_config.validate()
        self.logger = AgentLogger(verbose=verbose_logging, log_file=log_file)
        self.tools = self._create_tools(tavily_api_key)
        self.orchestrator = domain_config.orchestrator_class(
            llm, domain_config, self.logger
        )

        self.workflow = self._build_workflow()

    async def research(self, query: str) -> Dict[str, Any]:
        """Execute the complete research workflow with comprehensive logging"""

        self.logger.log(
            LogLevel.RESEARCH,
            "framework",
            "research_session_start",
            f"Starting research session for query: '{query}'",
            data={"domain": self.domain_config.domain_name},
        )

        try:
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
                logger=self.logger,
                reasoning_history=[],
            )

            app = self.workflow.compile()
            result = await app.ainvoke(initial_state)

            research_summary = self.logger.get_research_summary()

            return {
                "query": result["query"],
                "research_plan": result["research_plan"],
                "final_report": result["final_report"],
                "citations": result["citations"],
                "num_subagents": len(result["subagent_results"]),
                "errors": result["errors"],
                "execution_time": datetime.now().isoformat(),
                "research_summary": research_summary,
                "reasoning_chains": len(self.logger.reasoning_chains),
                "log_entries": len(self.logger.logs),
            }
        except Exception as e:
            self.logger.log(
                LogLevel.ERROR,
                "framework",
                "research_session_error",
                f"Error during research session: {str(e)}",
            )
            raise e

    def _create_tools(self, tavily_api_key: str) -> List[BaseTool]:
        """Create tools based on domain configuration"""
        tools = []

        if "web_search" in self.domain_config.tools:
            tools.append(TavilyWebSearchTool(tavily_api_key))

        if "intelligent_scraper" in self.domain_config.tools:
            tools.append(
                ScrapeGraphTool(
                    {
                        "model": "openai/gpt-4o",
                        "api_key": self.llm.openai_api_key,
                        "temperature": 0.1,
                    }
                )
            )

        if "citation_extractor" in self.domain_config.tools:
            tools.append(CitationTool())

        return tools

    def _build_workflow(self) -> StateGraph:
        """Build the research workflow"""
        workflow = StateGraph(ResearchState)

        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("execute_research", self._execute_research_node)
        workflow.add_node("synthesize_results", self._synthesize_results_node)
        workflow.add_node("add_citations", self._add_citations_node)

        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "execute_research")
        workflow.add_edge("execute_research", "synthesize_results")
        workflow.add_edge("synthesize_results", "add_citations")
        workflow.add_edge("add_citations", END)

        return workflow

    async def _analyze_query_node(self, state: ResearchState) -> ResearchState:
        """Analyze query and create research plan"""
        analysis = await self.orchestrator.analyze_query(state["query"])

        state["research_plan"] = analysis["research_plan"]
        state["subagent_tasks"] = analysis["subtasks"]
        state["current_step"] = "query_analyzed"

        return state

    async def _execute_research_node(self, state: ResearchState) -> ResearchState:
        """Execute research tasks using spawned agents and update state."""

        agents = []
        for task_data in state["subagent_tasks"]:
            specialist_class = self.domain_config.specialist_classes[0]
            agent_id = f"specialist_{task_data['id']}"
            specialist = specialist_class(
                self.llm, self.tools, self.domain_config, agent_id, self.logger
            )
            agents.append(specialist)

        async def execute_single_task(task_data, agent):
            task = SubagentTask(
                id=task_data["id"],
                objective=task_data["objective"],
                search_queries=task_data["search_queries"],
                tools_to_use=task_data["tools"],
                output_format=task_data["output_format"],
                boundaries=task_data["boundaries"],
            )
            return await agent.execute_task(task)

        task_pairs = list(zip(state["subagent_tasks"], agents))
        execution_tasks = [
            execute_single_task(task_data, agent) for task_data, agent in task_pairs
        ]
        results = await asyncio.gather(*execution_tasks, return_exceptions=True)

        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = (
                    f"Task {state['subagent_tasks'][i]['id']} failed: {str(result)}"
                )
                self.logger.log(
                    LogLevel.ERROR, "framework", "task_execution_error", error_msg
                )
                state["errors"].append(error_msg)
            else:
                valid_results.append(result)

        state["subagent_results"] = valid_results
        state["current_step"] = "research_completed"

        return state

    async def _synthesize_results_node(self, state: ResearchState) -> ResearchState:
        """Synthesize results into final report"""
        final_report = await self.orchestrator.synthesize_results(
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

        if "citation_extractor" in [tool.name for tool in self.tools]:
            citation_tool = next(
                tool for tool in self.tools if tool.name == "citation_extractor"
            )
            citation_result = citation_tool._run(state["final_report"], all_sources)
            state["citations"] = citation_result["citations"]
            state["final_report"] = citation_result["formatted_text"]

        state["current_step"] = "completed"
        return state
