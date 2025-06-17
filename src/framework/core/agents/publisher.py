from datetime import datetime
from typing import Any, Dict, List
from ..base import ResearchAgent
from ..logging import LogLevel


class PublisherAgent(ResearchAgent):
    """Publication formatting and export agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = self.research_config.workflow_prompts.publisher

    async def publish_report(
        self, report_content: str, citations: List[Dict], formats: List[str]
    ) -> Dict[str, Any]:
        """Format and prepare report for publication"""
        self.logger.log(
            LogLevel.AGENT,
            self.agent_id,
            "report_publishing",
            f"Publishing report in formats: {formats}",
        )

        metadata = {
            "title": self._extract_title(report_content),
            "publication_date": datetime.now().isoformat(),
            "word_count": len(report_content.split()),
            "citation_count": len(citations),
            "formats": formats,
        }

        return {
            "formatted_report": report_content,
            "metadata": metadata,
            "citations": citations,
            "publication_ready": True,
        }

    def _extract_title(self, content: str) -> str:
        """Extract title from report content"""
        lines = content.split("\n")
        for line in lines:
            if line.strip().startswith("#") and not line.startswith("##"):
                return line.replace("#", "").strip()
        return "Research Report"
