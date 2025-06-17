from ..logging import LogLevel
from ..base import ResearchAgent


class BrowserAgent(ResearchAgent):
    """Initial research and topic exploration agent"""

    def _build_system_prompt(self) -> str:
        return """You are a Research Browser Agent responsible for initial topic exploration.

Your role:
1. Conduct preliminary research to understand the topic scope
2. Identify key themes and areas for deeper investigation
3. Provide context for research planning
4. Use broad searches to map the research landscape

Focus on:
- Getting a comprehensive overview of the topic
- Identifying authoritative sources
- Understanding current developments and trends
- Noting potential research angles and subtopics"""

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

            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages(
                [("system", self.system_prompt), ("human", synthesis_prompt)]
            )

            response = await self.llm.ainvoke(prompt.format_messages())
            return response.content

        return f"Initial research overview for: {query}"
