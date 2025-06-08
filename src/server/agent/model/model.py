import os
from typing import Any, Dict, List

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage, ToolMessage)
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.server.agent.model.state import AgentState
from src.server.agent.prompt import agent_prompt
from src.server.agent.tools import ToolExecutor
from src.server.utils.cookie_utils import get_code_server_cookies


def initialize_llm():
    """Initialize the language model with proper configuration."""
    
    # Get environment variables with defaults
    model = os.getenv('AGENT_MODEL', 'gpt-4o')
    api_base = os.getenv('AGENT_MODEL_SERVER')
    api_key = os.getenv('AGENT_MODEL_API_KEY', "EMPTY")
    
    print(f"=== LLM Configuration ===")
    print(f"Model: {model}")
    print(f"API Base: {api_base}")
    print(f"API Key: {'SET' if api_key and api_key != 'EMPTY' else 'NOT SET'}")
    
    # Build extra headers
    extra_headers = {}
    try:
        cookies = get_code_server_cookies()
        if cookies and cookies.get('code-server-session'):
            extra_headers["Cookie"] = f"code-server-session={cookies.get('code-server-session')}"
    except Exception as e:
        print(f"Warning: Failed to get cookies: {e}")
    
    # Initialize with proper configuration
    llm_config = {
        "model": model,
        "streaming": False,  # Disable streaming for better response handling
        "temperature": 0.7,
        "max_tokens": 32768,  # Ensure we get complete responses
    }
    
    # Add API configuration if available
    if api_base:
        llm_config["openai_api_base"] = api_base
    if api_key:
        llm_config["openai_api_key"] = api_key
    if extra_headers:
        llm_config["extra_headers"] = extra_headers
    
    print(f"Final LLM config: {llm_config}")
    
    return ChatOpenAI(**llm_config)


def create_agent_workflow(_state_type=None):
    """
    Create and return a ReAct-style agent executor.

    The ReAct framework interleaves reasoning ("thought") and acting ("action")
    steps, letting the language model decide when to call a tool and when to
    answer the user. LangChain provides `create_react_agent`, which assembles this
    pattern from an LLM and a list of tools.
    """
    # Initialize the language model with the existing helper
    llm = initialize_llm()

    # Convert our custom ToolExecutor into LangChain‑compatible tools
    tool_executor = ToolExecutor()
    tools = tool_executor.as_tool()

    # Create system message for the agent
    from datetime import datetime

    from src.server.agent.config import PROJECT_ID
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system_message = SystemMessage(content=agent_prompt(PROJECT_ID, timestamp))

    # Build a ReAct agent with system message
    agent = create_react_agent(
        llm, 
        tools,
        state_modifier=system_message  # This ensures the system message is always included
    )
    return agent
