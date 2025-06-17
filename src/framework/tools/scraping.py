import json
from typing import Dict, Any
from langchain_core.tools import BaseTool
from scrapegraphai.graphs import SmartScraperGraph


class ScrapeGraphTool(BaseTool):
    """
    Web scraping tool that extracts specific information from web pages.
    Uses ScrapeGraph AI to understand content structure and extract relevant data.
    """

    name: str = "intelligent_scraper"
    description: str = """Extract specific information from web pages using AI-powered scraping.
    Provide a URL and describe what information you want to extract."""

    llm_config: Dict[str, Any]

    def __init__(self, llm_config: Dict[str, Any]):
        super().__init__(llm_config=llm_config)

    def _run(self, url: str, extraction_prompt: str) -> str:
        """Execute intelligent scraping on the given URL"""
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

    async def _arun(self, url: str, extraction_prompt: str) -> str:
        """Async version of the scraping"""
        return self._run(url, extraction_prompt)
