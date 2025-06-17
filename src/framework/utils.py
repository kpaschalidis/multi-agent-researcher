import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .core.types import ResearchConfig, WorkflowPrompts
from .research_workflow import ResearchWorkflow
from .core.logging import AgentLogger, LogLevel


@dataclass
class ResearchResult:
    """Standardized result from a ResearchOrchestrator run"""

    config_name: str
    query: str
    specialist_prompt: str
    final_report: str
    citations: List[Dict[str, str]]
    quality_score: float
    execution_time: str
    num_subagents: int
    errors: List[str]
    research_summary: Dict[str, Any]

    @classmethod
    def from_orchestrator_result(
        cls, config: ResearchConfig, result: Dict[str, Any]
    ) -> "ResearchResult":
        """Convert ResearchOrchestrator result to standardized format"""
        return cls(
            config_name=config.domain_name,
            query=result.get("query", ""),
            specialist_prompt=config.workflow_prompts.research_agent,
            final_report=result.get("final_report", ""),
            citations=result.get("citations", []),
            quality_score=result.get("quality_score", 0.0),
            execution_time=result.get("execution_time", ""),
            num_subagents=result.get("num_subagents", 0),
            errors=result.get("errors", []),
            research_summary=result.get("research_summary", {}),
        )


async def parallel_research(
    configs: List[ResearchConfig],
    base_query: str,
    llm: ChatOpenAI,
    tavily_api_key: str,
    verbose_logging: bool = True,
    log_file: Optional[str] = None,
) -> List[ResearchResult]:
    """
    Execute multiple ResearchOrchestrator runs in parallel

    Args:
        configs: List of research configurations for each parallel run
        base_query: Base query that will be combined with each config's focus
        llm: Language model instance
        tavily_api_key: API key for web search
        verbose_logging: Enable verbose logging
        log_file: Optional log file path

    Returns:
        List of ResearchResult objects from parallel execution
    """

    logger = AgentLogger(verbose=verbose_logging, log_file=log_file)

    logger.log(
        LogLevel.RESEARCH,
        "parallel_research",
        "parallel_execution_start",
        f"Starting parallel research with {len(configs)} configurations",
        data={
            "base_query": base_query,
            "configs": [{"name": c.name, "focus": c.research_focus} for c in configs],
            "execution_pattern": "parallel",
        },
    )

    start_time = datetime.now()

    # Create research tasks for parallel execution
    research_tasks = []

    for config in configs:
        query = f"{base_query} - {config.research_focus}"

        task = _execute_single_research(  # Use directly!
            config=config,
            query=query,
            llm=llm,
            tavily_api_key=tavily_api_key,
            logger=logger,
        )
        research_tasks.append(task)

    # Execute all research tasks in parallel
    logger.log(
        LogLevel.RESEARCH,
        "parallel_research",
        "executing_parallel_tasks",
        f"Executing {len(research_tasks)} research tasks in parallel",
    )

    results = await asyncio.gather(*research_tasks, return_exceptions=True)

    # Process results and handle exceptions
    research_results = []
    failed_configs = []

    for i, result in enumerate(results):
        config = configs[i]

        if isinstance(result, Exception):
            logger.log(
                LogLevel.ERROR,
                "parallel_research",
                "research_task_failed",
                f"Research task failed for {config.domain_name}: {str(result)}",
            )

            # Create failed result
            failed_result = ResearchResult(
                config_name=config.domain_name,
                query=f"{base_query} - {config.research_focus}",
                specialist_prompt=config.workflow_prompts.research_agent,
                final_report=f"Research failed: {str(result)}",
                citations=[],
                quality_score=0.0,
                execution_time=datetime.now().isoformat(),
                num_subagents=0,
                errors=[str(result)],
                research_summary={},
            )
            research_results.append(failed_result)
            failed_configs.append(config.domain_name)
        else:
            research_results.append(
                ResearchResult.from_orchestrator_result(config, result)
            )

    execution_time = datetime.now() - start_time

    logger.log(
        LogLevel.RESEARCH,
        "parallel_research",
        "parallel_execution_complete",
        f"Parallel research completed in {execution_time.total_seconds():.2f}s",
        data={
            "total_configs": len(configs),
            "successful_results": len(research_results) - len(failed_configs),
            "failed_configs": failed_configs,
            "execution_time_seconds": execution_time.total_seconds(),
        },
    )

    return research_results


async def sequential_research(
    configs: List[ResearchConfig],
    base_query: str,
    llm: ChatOpenAI,
    tavily_api_key: str,
    context_passing: bool = True,
    verbose_logging: bool = True,
    log_file: Optional[str] = None,
) -> List[ResearchResult]:
    """
    Execute multiple ResearchOrchestrator runs sequentially with optional context passing

    Args:
        configs: List of research configurations for sequential execution
        base_query: Base query that will be combined with each config's focus
        llm: Language model instance
        tavily_api_key: API key for web search
        context_passing: Whether to pass previous results as context to next phase
        verbose_logging: Enable verbose logging
        log_file: Optional log file path

    Returns:
        List of ResearchResult objects from sequential execution
    """

    logger = AgentLogger(verbose=verbose_logging, log_file=log_file)

    logger.log(
        LogLevel.RESEARCH,
        "sequential_research",
        "sequential_execution_start",
        f"Starting sequential research with {len(configs)} phases",
        data={
            "base_query": base_query,
            "configs": [{"name": c.name, "focus": c.research_focus} for c in configs],
            "context_passing": context_passing,
            "execution_pattern": "sequential",
        },
    )

    start_time = datetime.now()
    research_results = []
    accumulated_context = ""

    for i, config in enumerate(configs):
        phase_number = i + 1

        logger.log(
            LogLevel.RESEARCH,
            "sequential_research",
            "phase_start",
            f"Starting phase {phase_number}/{len(configs)}: {config.domain_name}",
            data={
                "phase_number": phase_number,
                "config_name": config.domain_name,
                "has_context": bool(accumulated_context),
            },
        )

        # Build query with context if enabled
        if context_passing and accumulated_context:
            phase_query = f"{base_query} - {config.research_focus}\n\nPrevious Research Context:\n{accumulated_context}"
        else:
            phase_query = f"{base_query} - {config.research_focus}"

        try:
            # Execute research for this phase
            result = await _execute_single_research(
                config=config,
                query=phase_query,
                llm=llm,
                tavily_api_key=tavily_api_key,
                logger=logger,
            )

            research_result = ResearchResult.from_orchestrator_result(config, result)
            research_results.append(research_result)

            # Add this result to context for next phase
            if context_passing:
                context_summary = (
                    f"\n{config.domain_name}: {research_result.final_report[:500]}..."
                )
                accumulated_context += context_summary

                # Trim context if it gets too long
                if len(accumulated_context) > 2000:
                    accumulated_context = accumulated_context[
                        -1500:
                    ]  # Keep last 1500 chars

            logger.log(
                LogLevel.RESEARCH,
                "sequential_research",
                "phase_complete",
                f"Phase {phase_number} completed successfully: {config.domain_name}",
                data={
                    "phase_number": phase_number,
                    "config_name": config.domain_name,
                    "quality_score": research_result.quality_score,
                    "context_length": len(accumulated_context),
                },
            )

        except Exception as e:
            logger.log(
                LogLevel.ERROR,
                "sequential_research",
                "phase_failed",
                f"Phase {phase_number} failed: {config.domain_name} - {str(e)}",
            )

            # Create failed result
            failed_result = ResearchResult(
                config_name=config.domain_name,
                query=phase_query,
                specialist_prompt=config.workflow_prompts.research_agent,
                final_report=f"Phase failed: {str(e)}",
                citations=[],
                quality_score=0.0,
                execution_time=datetime.now().isoformat(),
                num_subagents=0,
                errors=[str(e)],
                research_summary={},
            )
            research_results.append(failed_result)

            # Stop sequential execution on failure
            logger.log(
                LogLevel.ERROR,
                "sequential_research",
                "sequential_execution_stopped",
                f"Sequential execution stopped at phase {phase_number} due to failure",
            )
            break

    execution_time = datetime.now() - start_time

    logger.log(
        LogLevel.RESEARCH,
        "sequential_research",
        "sequential_execution_complete",
        f"Sequential research completed in {execution_time.total_seconds():.2f}s",
        data={
            "total_phases": len(configs),
            "completed_phases": len(research_results),
            "execution_time_seconds": execution_time.total_seconds(),
            "context_passing_used": context_passing,
        },
    )

    return research_results


async def synthesize_results(
    results: List[ResearchResult],
    synthesis_prompt: str,
    llm: ChatOpenAI,
    base_query: str = "",
    output_format: str = "Comprehensive Report",
    verbose_logging: bool = True,
) -> Dict[str, Any]:
    """
    Synthesize multiple research results into a unified report

    Args:
        results: List of ResearchResult objects to synthesize
        synthesis_prompt: System prompt for synthesis approach
        llm: Language model instance
        base_query: Original query being researched
        output_format: Desired output format
        verbose_logging: Enable verbose logging

    Returns:
        Dictionary containing synthesized report and metadata
    """

    logger = AgentLogger(verbose=verbose_logging)

    logger.log(
        LogLevel.RESEARCH,
        "synthesize_results",
        "synthesis_start",
        f"Starting synthesis of {len(results)} research results",
        data={
            "base_query": base_query,
            "result_configs": [r.config_name for r in results],
            "output_format": output_format,
        },
    )

    # Prepare research content for synthesis
    research_content = []
    all_citations = []
    total_quality_score = 0

    for result in results:
        if result.errors:
            # Include failed results with error context
            content = f"## {result.config_name.title().replace('_', ' ')}\n"
            content += f"**Status:** Research Failed\n"
            content += f"**Error:** {'; '.join(result.errors)}\n\n"
        else:
            content = f"## {result.config_name.title().replace('_', ' ')}\n"
            content += f"**Quality Score:** {result.quality_score:.2f}\n"
            content += f"**Research Focus:** {result.query.split(' - ')[-1] if ' - ' in result.query else result.query}\n\n"
            content += result.final_report + "\n\n"

            # Collect citations
            all_citations.extend(result.citations)
            total_quality_score += result.quality_score

        research_content.append(content)

    # Create synthesis prompt
    combined_content = "\n".join(research_content)

    synthesis_template = ChatPromptTemplate.from_messages(
        [
            ("system", synthesis_prompt),
            (
                "human",
                """Synthesize the following research results into a comprehensive report.

Original Query: {base_query}
Output Format: {output_format}

Research Results:
{research_content}

Create a synthesized report that:
1. Integrates insights from all research components
2. Identifies patterns and connections across findings  
3. Provides coherent analysis and recommendations
4. Notes any gaps or inconsistencies
5. Formats appropriately for the intended output format

Ensure the synthesis is more than just a summary - provide integrated analysis and strategic insights.""",
            ),
        ]
    )

    messages = synthesis_template.format_messages(
        base_query=base_query,
        output_format=output_format,
        research_content=combined_content,
    )

    try:
        synthesis_response = await llm.ainvoke(messages)
        synthesized_report = synthesis_response.content

        # Calculate synthesis metadata
        successful_results = [r for r in results if not r.errors]
        average_quality = (
            total_quality_score / len(successful_results) if successful_results else 0
        )

        synthesis_result = {
            "base_query": base_query,
            "synthesized_report": synthesized_report,
            "synthesis_metadata": {
                "total_components": len(results),
                "successful_components": len(successful_results),
                "failed_components": len(results) - len(successful_results),
                "average_quality_score": average_quality,
                "total_citations": len(all_citations),
                "output_format": output_format,
                "synthesis_timestamp": datetime.now().isoformat(),
            },
            "component_results": results,
            "all_citations": all_citations,
            "synthesis_quality": (
                "high"
                if average_quality > 0.8
                else "medium" if average_quality > 0.6 else "low"
            ),
        }

        logger.log(
            LogLevel.RESEARCH,
            "synthesize_results",
            "synthesis_complete",
            f"Synthesis completed successfully",
            data={
                "components_synthesized": len(results),
                "average_quality": average_quality,
                "output_length": len(synthesized_report),
                "synthesis_quality": synthesis_result["synthesis_quality"],
            },
        )

        return synthesis_result

    except Exception as e:
        logger.log(
            LogLevel.ERROR,
            "synthesize_results",
            "synthesis_failed",
            f"Synthesis failed: {str(e)}",
        )

        return {
            "base_query": base_query,
            "synthesized_report": f"Synthesis failed: {str(e)}",
            "synthesis_metadata": {
                "total_components": len(results),
                "synthesis_timestamp": datetime.now().isoformat(),
                "error": str(e),
            },
            "component_results": results,
            "all_citations": all_citations,
            "synthesis_quality": "failed",
        }


async def _execute_single_research(
    config: ResearchConfig,
    query: str,
    llm: ChatOpenAI,
    tavily_api_key: str,
    logger: AgentLogger,
) -> Dict[str, Any]:
    """Execute a single ResearchWorkflow run"""

    research_config = ResearchConfig(
        domain_name=config.domain_name,
        tools=config.tools,
        output_format=config.output_format,
        workflow_prompts=config.workflow_prompts or WorkflowPrompts(),
    )

    workflow = ResearchWorkflow(
        llm=llm,
        research_config=research_config,
        tavily_api_key=tavily_api_key,
        verbose_logging=logger.verbose,
        log_file=logger.log_file,
    )

    result = await workflow.research(query)

    return result
