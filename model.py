from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


def resultModel():
    model = ChatOpenAI(
        base_url=os.getenv("OPENROUTER_URL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="openrouter/free",
        max_tokens=1300,
        temperature=0.7,
    )
    return model
