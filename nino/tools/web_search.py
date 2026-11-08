"""Web search tool connecting to a self-hosted SearXNG instance with privacy sanitization."""

from typing import Any
import httpx
from pydantic import BaseModel, Field

from nino.security.privacy_gateway import PrivacyGateway, SecurityException
from nino.tools.base import BaseTool
from nino.tools.permissions import PermissionTier
from nino.tools.taint import TaintTracker


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Web search query string")
    max_results: int = Field(default=5, description="Maximum number of search results to return")


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web via self-hosted SearXNG instance. Queries are sanitized for privacy, "
        "and results are treated as untrusted data."
    )
    args_schema = WebSearchInput
    tier = PermissionTier.READ_ONLY

    def __init__(
        self,
        searxng_base_url: str = "http://127.0.0.1:8080",
        enabled: bool = False,
    ):
        self.searxng_base_url = searxng_base_url.rstrip("/")
        self.enabled = enabled

    async def run(self, query: str, max_results: int = 5) -> str:
        if not self.enabled:
            return (
                "Web search is currently disabled (ENABLE_WEB_SEARCH=false). "
                "To enable, configure a self-hosted SearXNG container and set ENABLE_WEB_SEARCH=true."
            )

        # 1. Anonymize outbound query through Privacy Gateway to scrub personal info, credentials, and conversational filler
        try:
            sanitized_query = PrivacyGateway.anonymize_search_query(query)
        except SecurityException as e:
            return f"Web search query blocked by Privacy Gateway: {e}"

        # 2. Validate endpoint URL
        url = f"{self.searxng_base_url}/search"
        params = {
            "q": sanitized_query,
            "format": "json",
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            return f"Failed to reach SearXNG instance at {self.searxng_base_url}: {e}"

        results = data.get("results", [])[:max_results]
        if not results:
            return f"No web search results found for '{sanitized_query}'."

        formatted_items = []
        for item in results:
            title = item.get("title", "No title")
            link = item.get("url", "")
            snippet = item.get("content", "")
            formatted_items.append(f"- **{title}**\n  URL: {link}\n  Snippet: {snippet}")

        raw_output = f"Web Search Results for '{sanitized_query}':\n\n" + "\n\n".join(formatted_items)
        return TaintTracker.wrap_untrusted(raw_output, source="searxng_web_search")
