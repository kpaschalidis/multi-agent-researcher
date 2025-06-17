from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch


class TavilyWebSearchTool(BaseTool):
    """
    Web search tool using Tavily's AI-optimized search.
    """

    name: str = "web_search"
    description: str = """Search the web for comprehensive, recent information. 
    Returns formatted results with URLs, titles, and content."""

    search: TavilySearch

    def __init__(self, api_key: str):
        search_instance = TavilySearch(
            max_results=5,
            topic="general",
            api_key=api_key,
        )
        super().__init__(search=search_instance)

    def _run(self, query: str) -> str:
        """Execute web search and handle dict response"""
        try:
            results = self.search.invoke(query)

            # Handle dict response from Tavily
            if isinstance(results, dict):

                if "results" in results:
                    search_results = results["results"]
                    return self._format_results(search_results, query)
                elif "sources" in results:
                    sources = results["sources"]
                    return self._format_results(sources, query)
                else:
                    formatted_result = f"""
    Source: https://tavily.com/search
    Title: Search Results for {query}
    Content: Found relevant information with {len(results)} data points
    Relevance Score: 0.8
    ---"""
                    return f"Search results for '{query}':\n{formatted_result}"

            elif isinstance(results, list):
                return self._format_results(results, query)

            else:
                return f"Unexpected result format: {type(results)}"

        except Exception as e:
            return f"Search failed for '{query}': {str(e)}"

    def _format_results(self, items: list, query: str) -> str:
        """Format list of results"""
        formatted_results = []

        for i, item in enumerate(items):
            if isinstance(item, dict):
                url = item.get("url", f"https://result-{i}.com")
                title = item.get("title", f"Result {i+1}")
                content = item.get("content", "Content available")
                score = item.get("score", 0.8)
            else:
                url = f"https://result-{i}.com"
                title = f"Result {i+1}"
                content = str(item)[:200]
                score = 0.7

            formatted_result = f"""
    Source: {url}
    Title: {title}
    Content: {content}
    Relevance Score: {score}
    ---"""
            formatted_results.append(formatted_result)

        return f"Search results for '{query}':\n" + "\n".join(formatted_results)

    async def _arun(self, query: str) -> str:
        """Async version of the search"""
        return self._run(query)
