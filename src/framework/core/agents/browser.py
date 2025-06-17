from langchain_core.prompts import ChatPromptTemplate

from ..logging import LogLevel
from ..base import ResearchAgent


class BrowserAgent(ResearchAgent):
    """Initial research and topic exploration agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = self.research_config.workflow_prompts.browser

    async def conduct_initial_research(self, query: str) -> str:
        """Conduct initial research to understand topic scope"""
        self.logger.log(
            LogLevel.AGENT,
            self.agent_id,
            "initial_research_start",
            f"Starting initial research for: {query}",
        )

        # Use web search to get broad overview
        if "web_search" in self.tools:
            search_results = self.tools["web_search"]._run(query)

            # Synthesize initial findings
            synthesis_prompt = f"""
            Based on this initial search about "{query}", provide a comprehensive overview including:
            1. Key themes and concepts
            2. Current state of the topic
            3. Major areas for deeper research
            4. Authoritative sources identified
            
            Search Results: {search_results}
            """

            prompt = ChatPromptTemplate.from_messages(
                [("system", self.system_prompt), ("human", synthesis_prompt)]
            )

            response = await self.llm.ainvoke(prompt.format_messages())
            return response.content

        return f"Initial research overview for: {query}"
