from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage


load_dotenv()

def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """

    print(f"searching for {query}")
    return "Tokyo weather is sunny"

def main():
    
    # Mistral has excellent tool calling support
    # Run: ollama pull mistral
    llm = ChatOllama(
        temperature=0, 
        model="mistral",
        num_predict=512,
    )
    
    tools = [search]
    agent = create_agent(model=llm, tools=tools)
    
    print("Invoking agent with Mistral...")
    response = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo")]})
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    main()
