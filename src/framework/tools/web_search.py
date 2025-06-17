from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch


class TavilyWebSearchTool(BaseTool):
    """
    Web search tool using Tavily's AI-optimized search.
    """

    name: str = "web_search"
    description: str = """Search the web for comprehensive, recent information. 
    Tavily provides AI-optimized search results with better source quality and relevance.
    Use broad queries first, then narrow down based on findings."""

    search: TavilySearch

    def __init__(self, api_key: str):
        search_instance = TavilySearch(
            max_results=5,
            topic="general",
            api_key=api_key,
        )
        super().__init__(search=search_instance)

    def _run(self, query: str) -> str:
        """Execute web search and format results"""
        try:
            results = self.search.invoke(query)
            formatted_results = []

            for result in results:
                formatted_result = f"""
Source: {result.get('url', 'Unknown')}
Title: {result.get('title', 'No title')}
Content: {result.get('content', 'No content')}
Relevance Score: {result.get('score', 'N/A')}
---"""
                formatted_results.append(formatted_result)

            return f"Search results for '{query}':\n" + "\n".join(formatted_results)

        except Exception as e:
            return f"Search failed for '{query}': {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async version of the search"""
        return self._run(query)
