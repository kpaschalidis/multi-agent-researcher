# Multi-Agent Research System

⚠️ Prototype

Research system built with LangGraph and LangChain, inspired by Anthropic's [multi-agent research architecture](https://www.anthropic.com/engineering/built-multi-agent-research-system). The system uses a lead researcher agent to coordinate multiple specialized subagents that work in parallel to gather and synthesize information.

## Features

- **Orchestrator-Worker Pattern**: Lead agent coordinates specialized subagents for parallel research
- **Intelligent Tool Selection**: Uses Tavily Search for web search and ScrapeGraph AI for intelligent web scraping
- **Adaptive Complexity**: Automatically scales the number of agents based on query complexity
- **Comprehensive Synthesis**: Combines findings from multiple sources into coherent reports
- **Citation Management**: Automatically tracks and formats source citations

## Architecture

```
User Query → Lead Researcher → Parallel Subagents → Synthesis → Final Report
                ↓                    ↓
        Research Planning    [Web Search + Scraping]
```

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key
- Tavily API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd multi-agent-research
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```
 
## Usage

### Basic Usage

```python
import asyncio
from multi_agent_researcher import MultiAgentResearcher
from langchain_openai import ChatOpenAI

async def main():
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create the research system
    researcher = MultiAgentResearcher(llm, os.getenv("TAVILY_API_KEY"))
    
    # Execute research
    result = await researcher.research("What are the latest developments in AI agent frameworks?")
    
    print("Research Plan:", result["research_plan"])
    print("Final Report:", result["final_report"])
    print("Citations:", len(result["citations"]))

if __name__ == "__main__":
    asyncio.run(main())
```

### Running the Example

```bash
python multi_agent_researcher.py
```

## Output Structure

```python
{
    "query": "Your research question",
    "research_plan": "Detailed plan created by lead agent",
    "final_report": "Comprehensive synthesized report",
    "citations": [{"id": "cite_1", "source": "...", "relevance_score": 0.8}],
    "num_subagents": 3,
    "errors": [],
    "execution_time": "2024-01-15T10:30:00"
}
```

## Configuration

Adjust system behavior by modifying parameters in the `MultiAgentResearcher` class:

```python
# Tavily search configuration
TavilyWebSearchTool(
    api_key=tavily_api_key,
    max_results=5,   
)

# ScrapeGraph configuration
ScrapeGraphTool({
    "model": "openai/gpt-4o",
    "temperature": 0.1,  # Control randomness in extraction
})
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

Based on the multi-agent research architecture described in [Anthropic's engineering blog post](https://www.anthropic.com/engineering/built-multi-agent-research-system).