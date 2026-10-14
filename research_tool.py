from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import tool
from web_reader import read_webpage

search = DuckDuckGoSearchResults(output_format="list", max_results=3)


@tool
def duckduckgo_search(query: str) -> list:
    """Use this tool to search the web for latest information.

    Input: a search query string
    Output: list of top search results with URLs

    Use this when the question requires:
    - latest news
    - current events
    - real-world facts
    - research topics"""

    print(">>> SEARCH TOOL CALLED")
    try:
        result = search.invoke(query)

        links = [r["link"] for r in result]

        pages = read_webpage.batch(links)

        return pages
    except Exception as e:
        return f"Error : {str(e)}"
