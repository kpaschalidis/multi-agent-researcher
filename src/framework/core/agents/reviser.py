from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
from ..base import ResearchAgent
from ..logging import LogLevel


class ReviserAgent(ResearchAgent):
    """Research refinement and improvement agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = self.research_config.workflow_prompts.reviser

    async def revise_research(
        self, research_data: List[Dict], feedback: List[Dict]
    ) -> List[Dict]:
        """Revise research based on reviewer feedback"""

        self.logger.log(
            LogLevel.AGENT,
            self.agent_id,
            "research_revision",
            f"Revising research based on {len(feedback)} feedback items",
        )

        revised_research = []

        for research_item in research_data:
            relevant_feedback = [
                f
                for f in feedback
                if f.get("section") == research_item["task_id"]
                or f.get("section") == "general"
            ]

            if relevant_feedback:
                revised_item = await self._apply_revisions(
                    research_item, relevant_feedback
                )
                revised_research.append(revised_item)
            else:
                revised_research.append(research_item)

        return revised_research

    async def _apply_revisions(self, research_item: Dict, feedback: List[Dict]) -> Dict:
        """Apply specific revisions to a research item"""

        feedback_text = "\n".join(
            [f"- {f['issue']}: {f['suggestion']}" for f in feedback]
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    """Revise this research findings based on the specific feedback provided.

Original Research:
Task: {task_id}
Objective: {objective}
Findings: {findings}

Reviewer Feedback:
{feedback}

Provide improved research that addresses all feedback points. Maintain the same structure but enhance content quality, accuracy, and completeness.""",
                ),
            ]
        )

        messages = prompt.format_messages(
            task_id=research_item["task_id"],
            objective=research_item["objective"],
            findings=research_item["findings"],
            feedback=feedback_text,
        )

        response = await self.llm.ainvoke(messages)

        return {
            **research_item,
            "findings": response.content,
            "revision_applied": True,
            "revision_feedback": feedback,
        }
