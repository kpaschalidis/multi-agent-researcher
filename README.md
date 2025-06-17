# Multi-Agent Research Framework 🔬🤖

Research system built with LangGraph and LangChain, inspired by Anthropic's [multi-agent research architecture](https://www.anthropic.com/engineering/built-multi-agent-research-system). The system uses a lead researcher agent to coordinate multiple specialized subagents that work in parallel to gather and synthesize information.

## Features

- **Multi-Agent Coordination**: Intelligent orchestrator spawns specialist agents based on query complexity
- **Logging**: See what agents think, decide, and do in real-time
- **Domain Specialization**: Easily extensible for different research domains
- **Tool Integration**: Web search, intelligent scraping, and citation management
- **Synthesis**: Combines findings from multiple sources into coherent reports
- **Citation Management**: Automatically tracks and formats source citations

## 🛠️ Installation

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

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from langchain_openai import ChatOpenAI
from framework import create_general_research_system

# Simple setup
llm = ChatOpenAI(model="gpt-4o-mini", api_key="your-openai-key")
research_system = create_general_research_system(
    llm=llm,
    tavily_api_key="your-tavily-key"
)

# Execute research
async def research():
    result = await research_system.research(
        "What are the latest developments in AI?"
    )
    print(result['final_report'])

asyncio.run(research())
```

### Advanced Usage with Custom Configuration

```python
from framework import (
    MultiAgentResearchFramework,
    GeneralResearchConfig,
    LogLevel
)

# Custom configuration
config = GeneralResearchConfig()
research_system = MultiAgentResearchFramework(
    llm=llm,
    domain_config=config,
    tavily_api_key="your-key",
    verbose_logging=True,
    log_file="research.log"
)
```

## 📝 Examples

### Run Basic Examples
```bash
python -m src.examples.demo
```

## 🔧 Extending the Framework

### Adding New Tools

```python
# Create in src/framework/tools/your_tool.py
from langchain_core.tools import BaseTool

class YourCustomTool(BaseTool):
    name = "your_tool"
    description = "Tool description"
    
    def _run(self, input: str) -> str:
        # Your tool implementation
        return result
```

### Creating Custom Domains

```python
# Create in src/framework/domains/your_domain.py
from ..core.base import BaseOrchestrator, BaseSpecialistAgent

class YourOrchestrator(BaseOrchestrator):
    def _build_system_prompt(self) -> str:
        return "Your domain-specific orchestrator prompt"

class YourSpecialist(BaseSpecialistAgent):
    def _build_system_prompt(self) -> str:
        return "Your domain-specific specialist prompt"
```
