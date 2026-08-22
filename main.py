from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from tavily import TavilyClient


load_dotenv()


tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over interet
    Args:
        query: The query to search for
    Returns:
        The search result
    """

    print(f"ssearching for {query}")
    return tavily.search(query=query)

def main():
    
    # llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    # llm = ChatOllama(temperature=0, model="gemma3:270m")  # This model doesn't support tools
    
    # Use llama3.1 which has better tool calling support than llama3.2
    # Make sure to run: ollama pull llama3.1
    llm = ChatOllama(
        temperature=0, 
        model="llama3.2",  # More reliable for tool calling than llama3.2
        num_predict=512,   # Ensure sufficient tokens for response
    )
    
    tools = [search]
    agent = create_agent(model=llm, tools=tools)
    
    print("Invoking agent...")
    response = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo")]})
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    main()
