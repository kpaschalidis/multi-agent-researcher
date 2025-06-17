from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool


class RequestsBeautifulSoupTool(BaseTool):
    """Scraping tool using Requests + BeautifulSoup"""

    name: str = "requests_scraper"
    description: str = "Extract content from web pages using Requests + BeautifulSoup"

    def _run(self, url: str, extraction_prompt: str = "") -> str:
        """Scrape webpage content with error handling"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = " ".join(chunk for chunk in chunks if chunk)

            # Limit content length to avoid token overflow
            if len(clean_text) > 8000:
                clean_text = clean_text[:8000] + "... [Content truncated]"

            return f"Successfully scraped from {url}:\n\nTitle: {soup.title.string if soup.title else 'No title'}\n\nContent:\n{clean_text}"

        except requests.exceptions.RequestException as e:
            return f"Request failed for {url}: {str(e)}"
        except Exception as e:
            return f"Scraping failed for {url}: {str(e)}"

    async def _arun(self, url: str, extraction_prompt: str = "") -> str:
        """Async version"""
        return self._run(url, extraction_prompt)


class PlaywrightScrapingTool(BaseTool):
    """Advanced scraping for JavaScript-heavy websites using Playwright"""

    name: str = "playwright_scraper"
    description: str = "Extract content from JavaScript-heavy websites using Playwright"
    playwright_available: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.playwright_available = self._check_playwright()

    def _check_playwright(self) -> bool:
        """Check if Playwright is available"""
        # We check if Playwright is available by importing it
        # This is a workaround to avoid Pydantic issues
        # TODO: Find a better way to check if Playwright is available
        try:
            from playwright.sync_api import sync_playwright

            return True
        except ImportError:
            print(
                "⚠️ Playwright not installed. Install with: pip install playwright && playwright install"
            )
            return False

    def _run(self, url: str, extraction_prompt: str = "") -> str:
        """Scrape using Playwright for JavaScript content"""
        if not self.playwright_available:
            return f"Playwright not available. Skipping {url}"

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()

                page.set_extra_http_headers(
                    {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                )

                page.goto(url, wait_until="networkidle", timeout=30000)
                page.wait_for_timeout(2000)

                title = page.title()
                content = page.evaluate(
                    """
                    () => {
                        // Remove scripts and styles
                        const scripts = document.querySelectorAll('script, style');
                        scripts.forEach(el => el.remove());
                        
                        const main = document.querySelector('main') || 
                                   document.querySelector('#main') || 
                                   document.querySelector('.main') ||
                                   document.querySelector('#content') ||
                                   document.body;
                        
                        return main ? main.innerText : document.body.innerText;
                    }
                """
                )

                browser.close()

                # Clean and limit content to avoid token overflow
                clean_content = " ".join(content.split())
                if len(clean_content) > 8000:
                    clean_content = clean_content[:8000] + "... [Content truncated]"

                return f"Successfully scraped from {url}:\n\nTitle: {title}\n\nContent:\n{clean_content}"

        except Exception as e:
            return f"Playwright scraping failed for {url}: {str(e)}"

    async def _arun(self, url: str, extraction_prompt: str = "") -> str:
        """Async version of the scraping tool"""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                await page.set_extra_http_headers(
                    {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                )

                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.wait_for_timeout(2000)

                title = await page.title()
                content = await page.evaluate(
                    """
                    () => {
                        const scripts = document.querySelectorAll('script, style');
                        scripts.forEach(el => el.remove());
                        
                        const main = document.querySelector('main') || 
                                   document.querySelector('#main') || 
                                   document.querySelector('.main') ||
                                   document.querySelector('#content') ||
                                   document.body;
                        
                        return main ? main.innerText : document.body.innerText;
                    }
                """
                )

                await browser.close()

                clean_content = " ".join(content.split())
                if len(clean_content) > 8000:
                    clean_content = clean_content[:8000] + "... [Content truncated]"

                return f"Successfully scraped from {url}:\n\nTitle: {title}\n\nContent:\n{clean_content}"

        except Exception as e:
            return f"Async Playwright scraping failed for {url}: {str(e)}"


class HybridScrapingTool(BaseTool):
    """Hybrid scraper that tries multiple methods

    The tool tries to scrape the content of a webpage using multiple methods.
    It first tries to use Requests + BeautifulSoup to scrape the content.
    If the content is insufficient, it tries to use Playwright to scrape the content.
    If the content is still insufficient, it returns the content that was scraped using Requests + BeautifulSoup.
    """

    name: str = "scraper"
    description: str = (
        "Intelligent scraper that tries multiple methods for best results"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_requests_scraper(self):
        """Get requests scraper instance"""
        return RequestsBeautifulSoupTool()

    def _get_playwright_scraper(self):
        """Get playwright scraper instance"""
        return PlaywrightScrapingTool()

    def _run(self, url: str, extraction_prompt: str = "") -> str:
        """Try multiple scraping methods for best results"""

        # First try simple requests approach
        requests_scraper = self._get_requests_scraper()
        result = requests_scraper._run(url, extraction_prompt)

        # Check if we got meaningful content
        if self._is_sufficient_content(result):
            return result

        # If content is insufficient, try Playwright for JavaScript content
        playwright_scraper = self._get_playwright_scraper()
        if playwright_scraper.playwright_available:
            playwright_result = playwright_scraper._run(url, extraction_prompt)
            if self._is_sufficient_content(playwright_result):
                return playwright_result

        return result

    async def _arun(self, url: str, extraction_prompt: str = "") -> str:
        """Async version with intelligent fallback"""

        # Try requests first
        requests_scraper = self._get_requests_scraper()
        result = await requests_scraper._arun(url, extraction_prompt)

        if self._is_sufficient_content(result):
            return result

        # Try Playwright if needed
        playwright_scraper = self._get_playwright_scraper()
        if playwright_scraper.playwright_available:
            playwright_result = await playwright_scraper._arun(url, extraction_prompt)
            if self._is_sufficient_content(playwright_result):
                return playwright_result

        return result

    def _is_sufficient_content(self, content: str) -> bool:
        """Check if scraped content is sufficient"""
        if "failed" in content.lower() or "error" in content.lower():
            return False

        if "Content:" in content:
            actual_content = content.split("Content:", 1)[1].strip()
            return len(actual_content) > 200

        return len(content) > 500


def create_scraping_tool(method: str = "hybrid") -> BaseTool:
    """
    Factory function to create scraping tools

    Args:
        method: "requests", "playwright", or "hybrid"

    Returns:
        Configured scraping tool
    """

    if method == "requests":
        return RequestsBeautifulSoupTool()

    elif method == "playwright":
        return PlaywrightScrapingTool()

    elif method == "hybrid":
        return HybridScrapingTool()

    else:
        return HybridScrapingTool()
