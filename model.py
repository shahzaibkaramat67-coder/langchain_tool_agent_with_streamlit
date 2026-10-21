from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from typing import Optional

load_dotenv()


def resultModel(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
):
    model = ChatOpenAI(
        base_url=base_url if base_url is not None else os.getenv("OPENROUTER_URL"),
        api_key=api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY"),
        model=model_name or os.getenv("OPENROUTER_MODEL", "openrouter/free"),
        max_tokens=1300,
        temperature=0.7,
    )
    return model
