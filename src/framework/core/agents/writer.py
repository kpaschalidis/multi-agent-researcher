from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
from ..base import ResearchAgent
from ..logging import LogLevel


class WriterAgent(ResearchAgent):
    """Report compilation and writing agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = self.research_config.workflow_prompts.writer

    async def compile_final_report(
        self, query: str, research_data: List[Dict], outline: Dict
    ) -> str:
        """Compile research into final publication-ready report"""
        self.logger.log(
            LogLevel.AGENT,
            self.agent_id,
            "report_compilation",
            f"Compiling final report for: {query}",
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    """Compile a comprehensive, publication-ready research report.

Research Query: {query}

Research Outline:
{outline}

Research Findings:
{research_data}

Create a professional report with:
1. Executive Summary
2. Introduction
3. Main Research Sections (based on outline)
4. Key Findings and Insights
5. Conclusions and Implications
6. Recommendations for Further Research

Use clear headings, maintain professional tone, and ensure logical flow between sections.""",
                ),
            ]
        )

        research_text = "\n\n".join(
            [f"## {r['objective']}\n{r['findings']}" for r in research_data]
        )

        messages = prompt.format_messages(
            query=query, outline=str(outline), research_data=research_text
        )

        response = await self.llm.ainvoke(messages)
        return response.content
