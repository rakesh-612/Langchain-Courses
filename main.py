from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ServerError

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()


# This decorator will automatically retry the function if a 503 ServerError occurs.
# It waits 2s, 4s, 8s, etc., up to 5 attempts before giving up.
@retry(
    retry=retry_if_exception_type(ServerError),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5)
)
def invoke_chain_with_retry(chain, info):
    return chain.invoke(input={"information": info})


def main():
    print("Hello from langchain-courses!")
    information = """
     Sachin Ramesh Tendulkar (born 24 April 1973) is an Indian former international cricketer who captained the Indian national team. Often dubbed the "God of Cricket" in India, he is widely regarded as one of the greatest cricketers of all time.[5] He holds several world records, including being the all-time highest run-scorer in international cricket,[6] receiving the most player of the match awards in international cricket,[7] and being the only batsman to score 100 international centuries.[8] Tendulkar was a Member of Parliament, Rajya Sabha by presidential nomination from 2012 to 2018.[9][10]

Tendulkar took up cricket at the age of eleven, made his Test match debut on 15 November 1989 against Pakistan in Karachi at the age of sixteen, and went on to represent Mumbai domestically and India internationally for over 24 years.[11] In 2002, halfway through his career, Wisden ranked him the second-greatest Test batsman of all time, behind Don Bradman, and the second-greatest ODI batsman of all time, behind Viv Richards.[12] The same year, Tendulkar was a part of the team that was one of the joint-winners of the 2002 ICC Champions Trophy. Later in his career, Tendulkar was part of the Indian team that won the 2011 Cricket World Cup, his first win in six World Cup appearances for India.[13] He had previously been named "Player of the Tournament" at the 2003 World Cup.

Tendulkar has received several awards from the government of India: the Arjuna Award (1994), the Khel Ratna Award (1997), the Padma Shri (1998), and the Padma Vibhushan (2008).[14][15] After Tendulkar played his last match in November 2013, the Prime Minister's Office announced the decision to award him the Bharat Ratna, India's highest civilian award.[16][17] He was the first sportsperson to receive the award and, as of 2024, is the youngest recipient.[18][19][20] Having retired from ODI cricket in 2012,[21][22] he retired from all forms of cricket in November 2013 after playing his 200th Test match.[23] Tendulkar played 664 international cricket matches in total, scoring 34,357 runs.[24] In 2013, Tendulkar was included in an all-time Test World XI to mark the 150th anniversary of Wisden Cricketers' Almanack, and he was one of only two specialist batsmen of the post–World War II era, along with Viv Richards, to get featured in the team.[25]

Tendulkar is regarded as a symbol of national pride in India for his achievements. In 2010, Time included Tendulkar in its annual list of the most influential people in the world.[26] Tendulkar was awarded the Sir Garfield Sobers Trophy for cricketer of the year at the 2010 International Cricket Council (ICC) Awards.[27] In 2019, he was inducted into the ICC Cricket Hall of Fame.[28]
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.0-flash", # You can revert to this or 2.0-flash
    #     temperature=0,
    #     google_api_key=os.getenv("GOOGLE_API_KEY")
    # )

    # llm = ChatGroq(
    #     model="llama-3.3-70b-versatile", # Groq's high-speed model
    #     temperature=0,
    #     api_key=os.getenv("GROQ_API_KEY") # Ensure this is in your .env
    # )

    llm = ChatOllama( model="gemma3:270m", temperature=0,)

    chain = summary_prompt_template | llm
    
    # Use the retry-enabled function
    try:
        response = invoke_chain_with_retry(chain, information)
        print(response)
    except Exception as e:
        print(f"Failed to get response after multiple retries: {e}")

if __name__ == "__main__":
    main()
