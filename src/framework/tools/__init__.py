"""
Tools for the Multi-Agent Research Framework
"""

from .web_search import TavilyWebSearchTool
from .scraping import (
    HybridScrapingTool,
    RequestsBeautifulSoupTool,
    PlaywrightScrapingTool,
)
from .citation import CitationTool

__all__ = [
    "TavilyWebSearchTool",
    "HybridScrapingTool",
    "RequestsBeautifulSoupTool",
    "PlaywrightScrapingTool",
    "CitationTool",
]
