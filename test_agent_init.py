"""
Test script for agent initialization.
This script verifies that the agent can be initialized without errors.
"""

import os
import sys
from agents.langchain_agents import ChatAssistantAgent, initialize_enhanced_agent

def test_agent_init():
    """Test that the agent initializes correctly"""
    print("\n--- Testing Agent Initialization ---")
    
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
        print("Will initialize in fallback mode")
    
    try:
        print("Initializing ChatAssistantAgent...")
        agent = ChatAssistantAgent(user_id="test_user")
        print("✓ ChatAssistantAgent initialized successfully")
        
        # Print which mode it's using
        if hasattr(agent, 'use_langchain') and agent.use_langchain:
            print("  Agent is using LangChain")
        else:
            print("  Agent is using simple fallback mode")
        
        return True
    except Exception as e:
        print(f"✗ Agent initialization failed: {e}")
        return False

if __name__ == "__main__":
    # Run the test
    success = test_agent_init()
    if success:
        print("\n✓ Agent initialization test passed!")
    else:
        print("\n✗ Agent initialization test failed!")
        sys.exit(1) 