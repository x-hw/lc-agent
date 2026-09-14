from langchain_tavily import TavilySearch


def make_search_tool(tavily_api_key: str):
    """Create the configured Tavily search tool when an API key exists."""
    if not tavily_api_key:
        return None
    return TavilySearch(
        tavily_api_key=tavily_api_key,
        max_results=5,
        search_depth="basic",
    )
