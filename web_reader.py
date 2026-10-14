import requests
from bs4 import BeautifulSoup
from langchain.tools import tool


@tool
def read_webpage(url: str) -> dict:
    """
    Extract clean text from a webpage URL.

    Args:
        url: Webpage URL to fetch content from

    Returns:
        dict containing cleaned text and source URL
    """
    print(">>> WEB READER CALLED")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        text = soup.get_text(separator=" ", strip=True)

        return {"url": url, "text": text[:5000]}

    except Exception as e:
        return {"error": str(e)}
