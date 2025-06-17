import asyncio
from typing import List, Dict, Any
from datetime import datetime

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from .tools.scraping import (
    PlaywrightScrapingTool,
    RequestsBeautifulSoupTool,
)

from .core.agents import (
    BrowserAgent,
    EditorAgent,
    ReviewerAgent,
    ReviserAgent,
    WriterAgent,
    PublisherAgent,
)
from .core.logging import AgentLogger, LogLevel
from .core.types import ResearchState, DomainConfig, SubagentTask
from .tools import TavilyWebSearchTool, CitationTool, HybridScrapingTool


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
            llm, domain_config, "orchestrator_agent", self.logger
        )

        self.workflow_agents = self._create_workflow_agents()
        self.workflow = self._build_workflow()

    async def research(
        self, query: str, publication_formats: List[str] = None
    ) -> Dict[str, Any]:
        """Execute the complete research workflow with comprehensive logging"""

        if publication_formats is None:
            publication_formats = ["markdown"]

        self.logger.log(
            LogLevel.RESEARCH,
            "framework",
            "research_session_start",
            f"Starting research session for query: '{query}'",
            data={
                "domain": self.domain_config.domain_name,
                "formats": publication_formats,
            },
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
                publication_formats=publication_formats,  # Properly set formats
                quality_guidelines=[],
                review_feedback=[],
                revision_history=[],
                quality_score=0.0,
                needs_revision=False,
            )

            app = self.workflow.compile()
            result = await app.ainvoke(initial_state)

            research_summary = self.logger.get_research_summary()

            return {
                "query": result["query"],
                "research_plan": result["research_plan"],
                "initial_research": result.get("initial_research", ""),
                "research_outline": result.get("research_outline", {}),
                "final_report": result["final_report"],
                "citations": result["citations"],
                "quality_score": result.get("quality_score", 0.0),
                "revision_history": result.get("revision_history", []),
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

    def _create_scraping_tool(self, method: str = "hybrid") -> BaseTool:
        """
        Factory function to create scraping tools

        Args:
            method: "requests", "playwright", or "hybrid"

        Returns:
            Configured scraping tool
        """

        if method == "requests":
            return RequestsBeautifulSoupTool()

        elif method == "playwright":
            return PlaywrightScrapingTool()

        else:
            return HybridScrapingTool()

    def _create_tools(self, tavily_api_key: str) -> List[BaseTool]:
        """Create tools based on domain configuration"""
        tools = []

        if "web_search" in self.domain_config.tools:
            tools.append(TavilyWebSearchTool(tavily_api_key))

        if "scraper" in self.domain_config.tools:
            tools.append(self._create_scraping_tool("hybrid"))

        if "citation_extractor" in self.domain_config.tools:
            tools.append(CitationTool())

        return tools

    def _create_workflow_agents(self) -> Dict[str, Any]:
        """Create specialized workflow agents"""
        return {
            "browser": BrowserAgent(
                self.llm, self.tools, self.domain_config, "browser_agent", self.logger
            ),
            "editor": EditorAgent(
                self.llm, self.domain_config, "editor_agent", self.logger
            ),
            "reviewer": ReviewerAgent(
                self.llm, self.tools, self.domain_config, "reviewer_agent", self.logger
            ),
            "reviser": ReviserAgent(
                self.llm, self.tools, self.domain_config, "reviser_agent", self.logger
            ),
            "writer": WriterAgent(
                self.llm, self.tools, self.domain_config, "writer_agent", self.logger
            ),
            "publisher": PublisherAgent(
                self.llm, self.tools, self.domain_config, "publisher_agent", self.logger
            ),
        }

    def _build_workflow(self) -> StateGraph:
        """Build the research workflow"""
        workflow = StateGraph(ResearchState)

        # Planning & Initial Research
        workflow.add_node("browser", self._browser_node)
        workflow.add_node("editor", self._editor_node)

        # Spawn Subagents
        workflow.add_node("execute_research", self._execute_research_node)

        # Quality Assurance Loop
        workflow.add_node("reviewer", self._reviewer_node)
        workflow.add_node("reviser", self._reviser_node)

        # Final Assembly
        workflow.add_node("writer", self._writer_node)
        workflow.add_node("publisher", self._publisher_node)

        # Workflow edges
        workflow.set_entry_point("browser")
        workflow.add_edge("browser", "editor")
        workflow.add_edge("editor", "execute_research")
        workflow.add_edge("execute_research", "reviewer")

        # Quality assurance loop
        workflow.add_conditional_edges(
            "reviewer",
            self._should_revise_or_continue,
            {
                "revise": "reviser",
                "writer": "writer",
                "force_continue": "writer",  # Force exit after max revisions
            },
        )
        workflow.add_edge("reviser", "reviewer")
        workflow.add_edge("writer", "publisher")
        workflow.add_edge("publisher", END)

        return workflow

    def _should_revise_or_continue(self, state: ResearchState) -> str:
        """
        Determine whether to revise or continue to writer
        Respects reviewer decisions while preventing infinite loops
        """
        # Get revision tracking
        revision_count = len(state.get("revision_history", []))
        max_revisions = 2  # Reduced from 3 - usually 1-2 revisions are sufficient

        # Get reviewer decision
        needs_revision = state.get("needs_revision", False)
        quality_score = state.get("quality_score", 0.8)

        self.logger.log(
            LogLevel.AGENT,
            "framework",
            "revision_decision_analysis",
            f"Analyzing revision decision: needs_revision={needs_revision}, "
            f"quality_score={quality_score:.2f}, revision_count={revision_count}",
            data={
                "needs_revision": needs_revision,
                "quality_score": quality_score,
                "revision_count": revision_count,
                "max_revisions": max_revisions,
            },
        )

        if revision_count >= max_revisions:
            self.logger.log(
                LogLevel.RESEARCH,
                "framework",
                "max_revisions_reached",
                f"Maximum revisions ({max_revisions}) reached. Proceeding to writer "
                f"with quality score {quality_score:.2f}",
                data={"final_decision": "proceed_to_writer", "reason": "max_revisions"},
            )
            return "writer"

        if needs_revision:
            self.logger.log(
                LogLevel.RESEARCH,
                "framework",
                "revision_approved",
                f"Reviewer requested revision (attempt {revision_count + 1}/{max_revisions}). "
                f"Quality score: {quality_score:.2f}",
                data={"final_decision": "revise", "reason": "reviewer_request"},
            )
            return "revise"

        self.logger.log(
            LogLevel.RESEARCH,
            "framework",
            "quality_approved",
            f"Research quality approved by reviewer. Quality score: {quality_score:.2f}. "
            f"Proceeding to writer.",
            data={"final_decision": "proceed_to_writer", "reason": "quality_approved"},
        )
        return "writer"

    async def _browser_node(self, state: ResearchState) -> ResearchState:
        """Initial research and topic exploration"""
        initial_research = await self.workflow_agents[
            "browser"
        ].conduct_initial_research(state["query"])
        state["initial_research"] = initial_research
        state["current_step"] = "initial_research_completed"
        return state

    async def _editor_node(self, state: ResearchState) -> ResearchState:
        """Research planning and task creation"""
        try:
            complexity = self.orchestrator.determine_complexity(state["query"])

            outline_result = await self.workflow_agents["editor"].plan_research_outline(
                state["query"],
                state["initial_research"],
                existing_complexity=complexity.value,
            )

            if "outline" in outline_result:
                state["research_outline"] = outline_result["outline"]
                state["quality_guidelines"] = outline_result.get(
                    "quality_guidelines", []
                )

                # Create subagent tasks based on outline
                analysis = await self.orchestrator.analyze_query(state["query"])
                state["research_plan"] = analysis["research_plan"]
                state["subagent_tasks"] = analysis["subtasks"]
            else:
                # Fallback to existing analysis
                analysis = await self.orchestrator.analyze_query(state["query"])
                state["research_plan"] = analysis["research_plan"]
                state["subagent_tasks"] = analysis["subtasks"]
                state["quality_guidelines"] = [
                    "Ensure accuracy",
                    "Use credible sources",
                    "Comprehensive coverage",
                ]

            state["current_step"] = "research_planned"

        except Exception as e:
            self.logger.log(
                LogLevel.ERROR,
                "editor_agent",
                "planning_failed",
                f"Research planning failed: {str(e)}",
            )
            # Fallback to basic analysis
            analysis = await self.orchestrator.analyze_query(state["query"])
            state["research_plan"] = analysis["research_plan"]
            state["subagent_tasks"] = analysis["subtasks"]
            state["quality_guidelines"] = ["Basic quality standards"]
            state["errors"].append(f"Editor agent error: {str(e)}")

        return state

    async def _execute_research_node(self, state: ResearchState) -> ResearchState:
        """Execute parallel research"""

        print(f"🔬 DEBUG: Execute research node starting")
        print(f"Current state keys: {list(state.keys())}")

        # Check if we have subagent tasks
        if not state.get("subagent_tasks"):
            print("❌ No subagent tasks found! Running analysis...")
            # Generate tasks using orchestrator
            analysis = await self.orchestrator.analyze_query(state["query"])
            state["research_plan"] = analysis["research_plan"]
            state["subagent_tasks"] = analysis["subtasks"]
            print(f"✅ Generated {len(state['subagent_tasks'])} subagent tasks")

        # Check domain config and specialist classes
        if not self.domain_config.specialist_classes:
            print("❌ No specialist classes configured!")
            state["errors"].append("No specialist classes available")
            state["subagent_results"] = []
            return state

        print(f"✅ Using specialist class: {self.domain_config.specialist_classes[0]}")

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

    async def _reviewer_node(self, state: ResearchState) -> ResearchState:
        """Quality review and validation"""
        review_result = await self.workflow_agents["reviewer"].review_research(
            state["subagent_results"], state["quality_guidelines"]
        )

        state["quality_score"] = review_result["overall_score"]
        state["needs_revision"] = review_result["needs_revision"]
        state["review_feedback"] = review_result.get("specific_feedback", [])
        state["current_step"] = "quality_reviewed"

        return state

    async def _reviser_node(self, state: ResearchState) -> ResearchState:
        """Research revision and improvement"""
        revised_results = await self.workflow_agents["reviser"].revise_research(
            state["subagent_results"], state["review_feedback"]
        )

        state["subagent_results"] = revised_results
        state["revision_history"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "feedback_addressed": len(state["review_feedback"]),
                "changes_made": "Research revised based on reviewer feedback",
            }
        )
        state["current_step"] = "research_revised"

        return state

    async def _writer_node(self, state: ResearchState) -> ResearchState:
        """Compile final report"""
        final_report = await self.workflow_agents["writer"].compile_final_report(
            state["query"], state["subagent_results"], state["research_outline"]
        )

        state["final_report"] = final_report
        state["current_step"] = "report_compiled"
        return state

    async def _publisher_node(self, state: ResearchState) -> ResearchState:
        """Format and prepare for publication with comprehensive citation debugging"""

        self.logger.log(
            LogLevel.AGENT,
            "publisher_agent",
            "publisher_start",
            "Starting publication process",
        )

        try:
            # Debug: Check what we have in subagent results
            self.logger.log(
                LogLevel.DEBUG,
                "publisher_agent",
                "subagent_results_check",
                f"Found {len(state['subagent_results'])} subagent results",
                data={
                    "subagent_results_count": len(state["subagent_results"]),
                    "subagent_results_keys": [
                        list(result.keys()) for result in state["subagent_results"]
                    ],
                },
            )

            # Collect all sources with detailed logging
            all_sources = []
            for i, result in enumerate(state["subagent_results"]):
                result_sources = result.get("sources", [])
                self.logger.log(
                    LogLevel.DEBUG,
                    "publisher_agent",
                    "collecting_sources",
                    f"Result {i} ({result.get('task_id', 'unknown')}): {len(result_sources)} sources",
                    data={
                        "task_id": result.get("task_id"),
                        "sources": result_sources,
                        "result_keys": list(result.keys()),
                    },
                )
                all_sources.extend(result_sources)

            self.logger.log(
                LogLevel.RESEARCH,
                "publisher_agent",
                "sources_collected",
                f"Total sources collected: {len(all_sources)}",
                data={"all_sources": all_sources},
            )

            # Initialize citations as empty list
            state["citations"] = []

            # Generate citations if tool is available and we have sources
            citation_tool = None
            for tool in self.tools:
                if hasattr(tool, "name") and tool.name == "citation_extractor":
                    citation_tool = tool
                    break

            if citation_tool and all_sources:
                self.logger.log(
                    LogLevel.AGENT,
                    "publisher_agent",
                    "generating_citations",
                    f"Generating citations for {len(all_sources)} sources using {citation_tool.name}",
                )

                try:
                    citation_result = citation_tool._run(
                        state["final_report"], all_sources
                    )

                    self.logger.log(
                        LogLevel.DEBUG,
                        "publisher_agent",
                        "citation_result",
                        f"Citation tool returned: {type(citation_result)}",
                        data={"citation_result": citation_result},
                    )

                    if (
                        isinstance(citation_result, dict)
                        and "citations" in citation_result
                    ):
                        state["citations"] = citation_result["citations"]

                        # Update report with formatted version if available
                        if "formatted_text" in citation_result:
                            state["final_report"] = citation_result["formatted_text"]

                        self.logger.log(
                            LogLevel.RESEARCH,
                            "publisher_agent",
                            "citations_generated",
                            f"Successfully generated {len(state['citations'])} citations",
                            data={"citations": state["citations"]},
                        )
                    else:
                        self.logger.log(
                            LogLevel.ERROR,
                            "publisher_agent",
                            "citation_format_error",
                            f"Citation tool returned unexpected format: {citation_result}",
                        )

                except Exception as citation_error:
                    self.logger.log(
                        LogLevel.ERROR,
                        "publisher_agent",
                        "citation_generation_error",
                        f"Citation generation failed: {str(citation_error)}",
                    )
                    # Continue without citations rather than failing

            elif not citation_tool:
                self.logger.log(
                    LogLevel.WARNING,
                    "publisher_agent",
                    "no_citation_tool",
                    "No citation_extractor tool found",
                )
            elif not all_sources:
                self.logger.log(
                    LogLevel.WARNING,
                    "publisher_agent",
                    "no_sources",
                    "No sources available for citation generation",
                )

            # Ensure publication formats are set
            if not state.get("publication_formats"):
                state["publication_formats"] = ["markdown"]

            self.logger.log(
                LogLevel.AGENT,
                "publisher_agent",
                "publishing_report",
                f"Publishing report in formats: {state['publication_formats']}",
                data={
                    "formats": state["publication_formats"],
                    "citations_count": len(state.get("citations", [])),
                    "report_length": len(state["final_report"]),
                },
            )

            # Publish report using the workflow agent
            if hasattr(self, "workflow_agents") and "publisher" in self.workflow_agents:
                publication_result = await self.workflow_agents[
                    "publisher"
                ].publish_report(
                    state["final_report"],
                    state.get("citations", []),
                    state["publication_formats"],
                )

                if (
                    isinstance(publication_result, dict)
                    and "formatted_report" in publication_result
                ):
                    state["final_report"] = publication_result["formatted_report"]

                self.logger.log(
                    LogLevel.RESEARCH,
                    "publisher_agent",
                    "publication_complete",
                    "Report publication completed successfully",
                )
            else:
                self.logger.log(
                    LogLevel.WARNING,
                    "publisher_agent",
                    "no_publisher_agent",
                    "No publisher workflow agent found, skipping advanced formatting",
                )

            state["current_step"] = "publication_ready"

        except Exception as e:
            error_msg = f"Publisher agent error: {str(e)}"
            self.logger.log(
                LogLevel.ERROR,
                "publisher_agent",
                "publishing_failed",
                f"Publishing failed: {str(e)}",
                data={"error": str(e), "error_type": type(e).__name__},
            )

            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_msg)

            # Ensure we still have basic citations list even on error
            if "citations" not in state:
                state["citations"] = []

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
