"""
Test script for memory implementation.
This script validates that the memory system is working correctly.
"""

import os
from pathlib import Path
import json
import time

# Try to import LangChain components
try:
    from langchain_openai import OpenAI
    from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
    from langchain_community.chat_message_histories import FileChatMessageHistory
    from langchain.chains import ConversationChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available. Install with: pip install langchain langchain-openai")
    exit(1)

# Import the memory manager and agent
from agents.langchain_agents import EnhancedMemoryManager, ChatAssistantAgent

def test_memory_persistence():
    """Test that memory persists between sessions"""
    print("\n--- Testing Memory Persistence ---")
    
    # Create a unique user ID for this test
    test_user_id = f"test_user_{int(time.time())}"
    print(f"Using test user ID: {test_user_id}")
    
    # Create memory manager
    memory_manager = EnhancedMemoryManager(user_id=test_user_id)
    
    # Add some test messages
    print("Adding test messages to memory...")
    memory_manager.add_user_message("Hello, this is a test message")
    memory_manager.add_ai_message("I received your test message")
    memory_manager.add_user_message("Can you remember this conversation?")
    memory_manager.add_ai_message("Yes, I'll remember this conversation")
    
    # Check that the file exists
    memory_file = memory_manager.memory_file
    print(f"Memory file: {memory_file}")
    
    if memory_file.exists():
        print(f"Memory file exists at {memory_file}")
        try:
            memory_data = json.loads(memory_file.read_text())
            print(f"Memory file contains {len(memory_data)} messages")
        except Exception as e:
            print(f"Error reading memory file: {e}")
    else:
        print("Memory file does not exist!")
        return False
    
    # Create a new memory manager with the same user ID to test persistence
    print("\nCreating new memory manager with same user ID...")
    new_memory_manager = EnhancedMemoryManager(user_id=test_user_id)
    
    # Get recent messages
    recent_messages = new_memory_manager.get_recent_messages()
    
    if recent_messages and len(recent_messages) == 4:
        print("✓ Memory persistence test passed!")
        print(f"Retrieved {len(recent_messages)} messages from persistent storage")
        return True
    else:
        print("✗ Memory persistence test failed!")
        print(f"Expected 4 messages, got {len(recent_messages) if recent_messages else 0}")
        return False

def test_agent_memory():
    """Test the agent's memory capabilities"""
    print("\n--- Testing Agent Memory ---")
    
    # Create a unique user ID for this test
    test_user_id = f"test_agent_{int(time.time())}"
    print(f"Using test user ID: {test_user_id}")
    
    # Create chat agent
    agent = ChatAssistantAgent(user_id=test_user_id)
    
    # Simulate a conversation
    print("Simulating a conversation...")
    
    responses = []
    
    # First message
    print("\nUser: Hello, my name is Test User")
    response1 = agent.generate_response("Hello, my name is Test User")
    print(f"Agent: {response1}")
    responses.append(response1)
    
    # Second message
    print("\nUser: What's my name?")
    response2 = agent.generate_response("What's my name?")
    print(f"Agent: {response2}")
    responses.append(response2)
    
    # Check if the second response contains the name
    name_remembered = "Test User" in response2
    if name_remembered:
        print("✓ Agent remembered the user's name!")
    else:
        print("✗ Agent failed to remember the user's name")
    
    # Get memory summary
    print("\nGetting memory summary...")
    summary = agent.get_memory_summary()
    print(f"Memory Summary: {summary}")
    
    return name_remembered

def main():
    """Run all memory tests"""
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key before running the tests")
        return
    
    print("Running memory tests...")
    
    # Test memory persistence
    persistence_result = test_memory_persistence()
    
    # Test agent memory
    agent_memory_result = test_agent_memory()
    
    # Overall result
    if persistence_result and agent_memory_result:
        print("\n✓ All memory tests passed!")
    else:
        print("\n✗ Some memory tests failed")
        if not persistence_result:
            print("  - Memory persistence test failed")
        if not agent_memory_result:
            print("  - Agent memory test failed")

if __name__ == "__main__":
    main() 