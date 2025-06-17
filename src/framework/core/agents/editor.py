from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
import json5

from ..base import ResearchLead
from ..logging import LogLevel
from ..types import TaskComplexity


class EditorAgent(ResearchLead):
    """Research planning and coordination agent"""

    def _build_system_prompt(self) -> str:
        return """You are a Research Editor Agent responsible for planning and structuring research.

Your responsibilities:
1. Analyze initial research to understand scope
2. Create detailed research outlines with clear sections
3. Plan parallel research tasks for optimal coverage
4. Define quality guidelines for the research
5. Coordinate the overall research workflow

Focus on:
- Creating comprehensive research plans
- Ensuring logical flow and coverage
- Optimizing parallel research efficiency
- Setting quality standards and guidelines"""

    async def plan_research_outline(
        self, query: str, initial_research: str, existing_complexity: str = None
    ) -> Dict[str, Any]:
        """Create research outline using existing complexity analysis"""

        self.logger.log(
            LogLevel.AGENT,
            self.agent_id,
            "outline_planning",
            f"Planning research outline for: {query}",
        )

        # Use provided complexity instead of re-analyzing
        if existing_complexity:
            complexity = TaskComplexity(existing_complexity)
            self.logger.log(
                LogLevel.AGENT,
                self.agent_id,
                "using_cached_complexity",
                f"Using existing complexity analysis: {complexity.value}",
            )
        else:
            complexity = self.determine_complexity(query)

        config = self.get_agent_config(complexity)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    """Create a comprehensive research outline based on the initial research.

Query: {query}
Initial Research: {initial_research}
Complexity: {complexity}
Number of Research Tasks: {num_tasks}

Create a detailed outline with:
1. Research sections and subtopics
2. Quality guidelines for each section
3. Success criteria for completion
4. Research priorities and dependencies

Return as JSON with structure:
{{
    "outline": {{
        "main_sections": ["section1", "section2", ...],
        "section_details": {{"section1": {{"objective": "...", "key_questions": [...], "sources_needed": [...]}}}},
        "research_flow": "description of research approach"
    }},
    "quality_guidelines": [
        "guideline1: specific quality requirement",
        "guideline2: source credibility standards",
        ...
    ],
    "success_criteria": {{
        "completeness": "definition of complete research",
        "accuracy": "accuracy standards", 
        "depth": "depth requirements"
    }}
}}""",
                ),
            ]
        )

        messages = prompt.format_messages(
            query=query,
            initial_research=initial_research,
            complexity=complexity.value,
            num_tasks=config["num_subagents"],
        )

        response = await self.llm.ainvoke(messages)

        try:
            result = json5.loads(response.content.strip())
            return result
        except:
            # Fallback to simpler analysis
            return {
                "outline": {
                    "main_sections": [
                        "overview",
                        "current_developments",
                        "tools_and_frameworks",
                    ]
                },
                "quality_guidelines": [
                    "Ensure accuracy",
                    "Use credible sources",
                    "Provide comprehensive coverage",
                ],
                "success_criteria": {
                    "completeness": "Cover all main aspects",
                    "accuracy": "Fact-checked",
                    "depth": "Detailed analysis",
                },
            }
