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
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

class SourceModel(BaseModel):
    """ Schema for a source used by the agent """
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """ Schema for agent response with answer and sources """
    answer:str = Field(description="The agent answer's to the query")
    sources: List[SourceModel] = Field(default_factory=list, description="List of sources used to generate the answer")

llm = ChatGroq(model="llama-3.3-70b-versatile")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-courses!")
    result = agent.invoke({
        "messages": [
            HumanMessage(content="Find 3 frontend developer jobs using LangChain in the Chennai Area")
        ]
    })
    print(result)


if __name__ == "__main__":
    main()
