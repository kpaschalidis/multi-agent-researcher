from typing import List, Dict, Any
from langchain_core.tools import BaseTool


class CitationTool(BaseTool):
    """
    Tool for extracting and formatting citations from research results.
    """

    name: str = "citation_extractor"
    description: str = (
        "Extract and format citations from research results with proper academic formatting"
    )

    def _run(self, text: str, sources: List[str]) -> Dict[str, Any]:
        """Extract citations and format the text with proper references"""

        citations = []
        formatted_text = text

        if not sources:
            return {
                "citations": [],
                "formatted_text": text,
                "citation_count": 0,
            }

        for i, source in enumerate(sources, 1):
            if ": " in source:
                source_url = source.split(": ", 1)[1]
            else:
                source_url = source

            citation = {
                "id": f"ref_{i}",
                "source": source_url,
            }
            citations.append(citation)

        if citations:
            formatted_text += "\n\n## References\n\n"
            for i, citation in enumerate(citations, 1):
                source_url = citation["source"]
                formatted_text += f"{i}. {source_url}\n"

        return {
            "citations": citations,
            "formatted_text": formatted_text,
            "citation_count": len(citations),
        }

    async def _arun(self, text: str, sources: List[str]) -> Dict[str, Any]:
        """Async version of citation extraction"""
        return self._run(text, sources)
