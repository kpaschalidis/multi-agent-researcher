from datetime import datetime
from typing import Any, Dict, List
from ..base import BaseSpecialistAgent
from ..logging import LogLevel


class PublisherAgent(BaseSpecialistAgent):
    """Publication formatting and export agent"""

    def _build_system_prompt(self) -> str:
        return """You are a Research Publisher Agent responsible for final publication formatting.

Your role:
1. Format reports for multiple output types
2. Ensure proper citation formatting
3. Create publication metadata
4. Handle export and distribution
5. Maintain formatting standards

Focus on:
- Professional formatting standards
- Multiple format compatibility
- Proper citation styles
- Metadata completeness
- Publication readiness"""

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
