from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
# from tavily import TavilyClient

from langchain_tavily import TavilySearch
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


# tavily = TavilyClient()


# @tool
# def weather_search(query: str) -> str:
#     """
#     Use this tool to get current weather information for any city.
#     Always use this tool for weather-related questions.
#     """
#     print(f"Searching for {query}")

#     response = tavily.search(query=f"current weather in {query}")

#     if response.get("answer"):
#         return response["answer"]

#     if response.get("results"):
#         content = response["results"][0].get("content", "")

#         return f"Weather info for {query}: {content}"

#     return "Weather information not available."


llm = ChatGroq(model="llama-3.1-8b-instant")
# tools = [weather_search]

# tools = [TavilySearch()]
tools = [TavilySearchResults()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-courses!")
    result = agent.invoke({
        "messages": [
            # SystemMessage(content="""
            #   You must use the weather_search tool once.
            #   After getting the result, return the final answer.
            #   Do NOT call the tool again.
            # """),

            # HumanMessage(content="What is the weather in Tokyo?")

            HumanMessage(content="Find 3 frontend developer jobs using LangChain in the Bay Area")
        ]
    })
    print(result)


if __name__ == "__main__":
    main()
