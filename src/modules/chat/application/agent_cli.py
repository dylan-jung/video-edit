import json
from datetime import datetime

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage, ToolMessage)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.modules.chat.config import BASE_PATH, PROJECT_ID
from .workflow import create_agent_workflow
from src.modules.chat.domain.state import AgentState
from .prompt import agent_prompt
from src.modules.chat.infrastructure.tools import *


def run_agent():
    """Runs the agent in a loop, similar to the previous implementation."""
    agent_graph = create_agent_workflow(AgentState, project_id=PROJECT_ID)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Initialize with system message
    messages = [SystemMessage(content=agent_prompt(PROJECT_ID, timestamp))]
    
    while True:
        try:
            # Get user input
            user_input = input("\nUser ▶ ")
            
            # Add user message to history
            messages.append(HumanMessage(content=user_input))
            
            # Run the agent
            result = agent_graph.invoke({"messages": messages, "project_id": PROJECT_ID, "timestamp": timestamp})
            
            # Update messages with the result
            messages = result["messages"]
            
            # Find the last assistant message that isn't a tool call
            assistant_message = None
            
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                        assistant_message = msg
                        break
            
            if assistant_message:
                # Try to extract reasoning from the content (some models put it in the content)
                content = assistant_message.content
                reasoning = None
                
                # Check if we have a JSON structure that might contain reasoning
                try:
                    if content.startswith('{') and content.endswith('}'):
                        content_json = json.loads(content)
                        if 'reasoning' in content_json:
                            reasoning = content_json['reasoning']
                            content = content_json.get('response', content)
                except:
                    pass
                
                # Print reasoning if found
                if reasoning:
                    print("Assistant Reasoning ▶", reasoning)
                
                # Print the final response
                print("Assistant ▶", content)
            else:
                print("Assistant ▶ No response generated.")
        
        except KeyboardInterrupt:
            print(f"messages: {messages[1:]}")
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    run_agent()
