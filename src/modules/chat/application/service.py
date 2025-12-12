import logging
import json
import re
from datetime import datetime
from typing import List, Dict, Any, AsyncGenerator

import os
from langchain_core.messages import HumanMessage
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from src.modules.chat.config import PROJECT_ID
from src.modules.chat.application.workflow import create_agent_workflow
from src.modules.chat.domain.state import AgentState
from src.modules.chat.application.prompt import agent_prompt

logger = logging.getLogger(__name__)

# Default to localhost for local dev, override in production
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

class AgentService:
    def __init__(self, project_id: str = PROJECT_ID):
        self.project_id = project_id
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Note: Graph with checkpointer is created per-request/connection logic usually
        # but we can initialize the base graph components here or just lazily.

    async def chat(self, user_input: str, thread_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Processes user input and yields streaming responses (tokens or status updates).
        """
        try:
            # Setup checkpointer
            client = MongoClient(MONGO_URI)
            checkpointer = MongoDBSaver(client)
            
            # Create graph with checkpointer
            agent_graph = create_agent_workflow(AgentState, project_id=self.project_id, checkpointer=checkpointer)
            
            config = {"configurable": {"thread_id": thread_id}}
            
            # Stream the execution
            async for event in agent_graph.astream_events(
                {"messages": [HumanMessage(content=user_input)], "project_id": self.project_id, "timestamp": self.timestamp},
                config,
                version="v2"
            ):
                kind = event["event"]
                
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        yield {"type": "token", "content": content}
                        
                elif kind == "on_tool_start":
                     yield {"type": "info", "content": f"Starting tool: {event['name']}"}
                     
                elif kind == "on_tool_end":
                     yield {"type": "info", "content": f"Tool finished: {event['name']}"}

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            yield {"type": "error", "content": str(e)}

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            yield {"type": "error", "content": str(e)}

    def _extract_thinking(self, raw_content: str) -> str:
        """Extracts content within <think> tags."""
        import re
        thinking_parts = []
        think_matches = re.findall(r'<think>(.*?)</think>', raw_content, flags=re.DOTALL)
        for match in think_matches:
            thinking_parts.append(match.strip())
        
        if not think_matches:
             unclosed_match = re.search(r'<think>(.*)', raw_content, flags=re.DOTALL)
             if unclosed_match:
                 thinking_parts.append(unclosed_match.group(1).strip())

        return '\n\n'.join(thinking_parts).strip()

    def _clean_ai_response(self, raw_content: str) -> str:
        """Removes <think> tags to get the final answer."""
        import re
        content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL)
        content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
        return content.strip()
