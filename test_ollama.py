"""Test script to verify Ollama connection and model availability"""
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

def test_tool(query: str) -> str:
    """Test tool for agent"""
    return f"Test result for: {query}"

# Test different models with tool calling
models_to_test = [
    "llama3.1",
    "mistral",
    "qwen2.5",
    "llama3.2",
]

print("Testing Ollama models with tool calling support...\n")

for model_name in models_to_test:
    try:
        print(f"Testing {model_name}...")
        llm = ChatOllama(
            model=model_name,
            temperature=0,
            num_predict=512,
        )
        
        # Test basic invocation
        response = llm.invoke("Say 'OK'")
        print(f"  Basic test: {response.content[:50]}...")
        
        # Test with agent/tools
        agent = create_agent(model=llm, tools=[test_tool])
        result = agent.invoke({"messages": [HumanMessage(content="Use the test tool")]})
        
        print(f"✓ {model_name} works with tool calling!\n")
        print(f"Recommended: Use '{model_name}' in your main.py\n")
        break  # Stop after first working model
        
    except Exception as e:
        print(f"✗ {model_name} failed: {str(e)[:100]}...\n")

print("Test complete.")
