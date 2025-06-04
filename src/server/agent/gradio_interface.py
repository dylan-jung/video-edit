import json
from datetime import datetime
from typing import List, Tuple

import gradio as gr
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage, ToolMessage)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.server.agent.config import BASE_PATH, PROJECT_ID
from src.server.agent.model.model import create_agent_workflow
from src.server.agent.model.state import AgentState
from src.server.agent.prompt import agent_prompt
from src.server.agent.tools import *


class AgentInterface:
    def __init__(self):
        """Initialize the agent interface with workflow and message history."""
        self.agent_graph = create_agent_workflow(AgentState)
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Initialize with system message
        self.messages = [SystemMessage(content=agent_prompt(PROJECT_ID, self.timestamp))]
        
    def reset_conversation(self):
        """Reset the conversation by clearing message history."""
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.messages = [SystemMessage(content=agent_prompt(PROJECT_ID, self.timestamp))]
        return []
        
    def chat_with_agent(self, user_input: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
        """
        Process user input and return updated chat history.
        
        Args:
            user_input: User's message
            history: Chat history as list of (user_msg, assistant_msg) tuples
            
        Returns:
            Updated history and empty string to clear input
        """
        if not user_input.strip():
            return history, ""
            
        try:
            # Add user message to internal history
            self.messages.append(HumanMessage(content=user_input))
            
            # Run the agent
            result = self.agent_graph.invoke({
                "messages": self.messages, 
                "project_id": PROJECT_ID, 
                "timestamp": self.timestamp
            })
            
            # Update messages with the result
            self.messages = result["messages"]
            
            # Find the last assistant message that isn't a tool call
            assistant_message = None
            
            for msg in reversed(self.messages):
                if isinstance(msg, AIMessage):
                    if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                        assistant_message = msg
                        break
            
            if assistant_message:
                # Try to extract reasoning from the content
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
                
                # Prepare the response
                response = content
                if reasoning:
                    response = f"**Reasoning:** {reasoning}\n\n**Response:** {content}"
                
                # Update chat history
                history.append((user_input, response))
            else:
                history.append((user_input, "No response generated."))
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history.append((user_input, error_msg))
        
        return history, ""


def create_gradio_interface():
    """Create and configure the Gradio interface."""
    
    # Initialize agent interface
    agent_interface = AgentInterface()
    
    # Create Gradio interface
    with gr.Blocks(
        title="Attenz AI Agent",
        theme=gr.themes.Soft(),
        css="""
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .chat-container {
            height: 600px;
        }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # 🤖 Attenz AI Agent
            
            Welcome to Attenz AI Agent! I can help you with various tasks including:
            - 📹 Video analysis and scene understanding
            - 🔍 Searching through your video database  
            - 📊 Data analysis and insights
            - 💬 General conversation and assistance
            
            Start chatting below!
            """
        )
        
        with gr.Row():
            with gr.Column(scale=4):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Chat with Agent",
                    height=500,
                    show_label=True,
                    elem_classes=["chat-container"],
                    bubble_full_width=False
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your message here...",
                        label="Message",
                        lines=2,
                        max_lines=5,
                        show_label=False,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
                    
            with gr.Column(scale=1):
                # Info panel
                gr.Markdown(
                    """
                    ### 💡 Tips
                    
                    **Video Analysis:**
                    - "Analyze the scene in video X"
                    - "What objects are in this video?"
                    
                    **Search:**
                    - "Find videos with cars"
                    - "Search for outdoor scenes"
                    
                    **General:**
                    - "What can you help me with?"
                    - "Show me your capabilities"
                    """
                )
                
                gr.Markdown(
                    f"""
                    ### 📋 Session Info
                    
                    **Project ID:** {PROJECT_ID}
                    **Session Started:** {agent_interface.timestamp}
                    """
                )
        
        # Event handlers
        def handle_submit(user_input, history):
            return agent_interface.chat_with_agent(user_input, history)
            
        def handle_clear():
            agent_interface.reset_conversation()
            return []
        
        # Connect events
        msg.submit(
            handle_submit,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],
            show_progress=True
        )
        
        send_btn.click(
            handle_submit,
            inputs=[msg, chatbot], 
            outputs=[chatbot, msg],
            show_progress=True
        )
        
        clear_btn.click(
            handle_clear,
            outputs=[chatbot],
            show_progress=False
        )
        
        # Example messages
        examples = [
            "What can you help me with?",
            "Analyze a video scene",
            "Search for videos containing specific objects",
            "Show me your available tools"
        ]
        
        gr.Examples(
            examples=examples,
            inputs=msg,
            label="💬 Example Prompts"
        )
    
    return demo


def main():
    """Launch the Gradio interface."""
    demo = create_gradio_interface()
    
    print("🚀 Starting Attenz AI Agent Interface...")
    print("📱 Access the interface at: http://localhost:7860")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_api=True,
        enable_monitoring=True,
        max_threads=10
    )


if __name__ == "__main__":
    main() 