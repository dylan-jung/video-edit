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
        print("🔄 Initializing Agent Interface...")
        try:
            self.agent_graph = create_agent_workflow(AgentState)
            print("✅ Agent workflow created successfully")
            
            
        except Exception as e:
            print(f"❌ Failed to create agent workflow: {e}")
            import traceback
            traceback.print_exc()
            raise e
            
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Initialize with system message
        self.messages = [SystemMessage(content=agent_prompt(PROJECT_ID, self.timestamp))]
        print(f"✅ Agent initialized with {len(self.messages)} initial messages")
        
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
            print(f"🔵 User input: {user_input}")
            
            # Add user message to internal history
            self.messages.append(HumanMessage(content=user_input))
            print(f"🔵 Messages before agent call: {len(self.messages)}")
            
            # Run the agent
            result = self.agent_graph.invoke({
                "messages": self.messages, 
                "project_id": PROJECT_ID, 
                "timestamp": self.timestamp
            })
            
            print(f"🔵 Messages after agent call: {len(result['messages'])}")
            
            # Update messages with the result
            self.messages = result["messages"]
            
            # Debug: Print all messages to understand the structure
            print("🔍 All messages after agent call:")
            for i, msg in enumerate(self.messages):
                msg_type = type(msg).__name__
                content_length = len(msg.content) if hasattr(msg, 'content') and msg.content else 0
                has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
                print(f"  {i}: {msg_type} - Content: {content_length} chars, Tool calls: {has_tool_calls}")
                if hasattr(msg, 'content') and msg.content:
                    print(f"     Content: {msg.content[:100]}...")
            
            # Simple approach: find the last AI message with content
            response_content = ""
            
            # Go through messages from newest to oldest
            for msg in reversed(self.messages):
                if isinstance(msg, AIMessage):
                    print(f"🔍 Found AIMessage with content: {bool(msg.content)}")
                    if msg.content and msg.content.strip():
                        raw_content = msg.content.strip()
                        
                        # Extract thinking and clean response separately
                        thinking_content = self._extract_thinking(raw_content)
                        cleaned_content = self._clean_ai_response(raw_content)
                        
                        print(f"🧠 Thinking extracted: {len(thinking_content)} chars")
                        print(f"💬 Response extracted: {len(cleaned_content)} chars")
                        
                        # Format the final response with thinking
                        response_content = self._format_response_with_thinking(thinking_content, cleaned_content)
                        
                        print(f"✅ Using formatted response: {response_content[:100]}...")
                        break
            
            # If no AI message with content, try to create a helpful response
            if not response_content:
                print("🔍 No AI content found, checking for tool usage...")
                
                # Look for tool calls in AI messages
                for msg in reversed(self.messages):
                    if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                        tool_names = [call.get('name', 'unknown') for call in msg.tool_calls]
                        response_content = f"I used the following tools to help you: {', '.join(tool_names)}. The operation was completed successfully."
                        print(f"✅ Created tool response: {response_content}")
                        break
            
            # Final fallback - at least acknowledge the user
            if not response_content:
                response_content = "I received your message and I'm here to help! However, I didn't generate a specific response. Could you please try rephrasing your question or let me know what you'd like me to help you with?"
                print("⚠️ Using acknowledgment fallback")
            
            print(f"🔵 Final response: {response_content[:100]}...")
            history.append((user_input, response_content))
                
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error while processing your request: {str(e)}"
            print(f"❌ Exception in chat_with_agent: {e}")
            import traceback
            traceback.print_exc()
            history.append((user_input, error_msg))
        
        print(f"🔵 Final history length: {len(history)}")
        return history, ""

    def _clean_ai_response(self, raw_content: str) -> str:
        """
        Clean AI response by removing internal thinking tags and extracting user-facing content.
        
        Args:
            raw_content: Raw AI response content
            
        Returns:
            Cleaned response content suitable for display
        """
        import re

        # Remove <think> tags and their content
        content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL)
        
        # Remove any remaining think tags (unclosed)
        content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
        
        # Remove other common internal tags
        content = re.sub(r'</?thinking>', '', content, flags=re.IGNORECASE)
        content = re.sub(r'</?internal>', '', content, flags=re.IGNORECASE)
        
        # Clean up whitespace
        content = content.strip()
        
        # If content is empty after cleaning, try to extract from original
        if not content:
            # Look for content after closing think tag
            think_match = re.search(r'</think>\s*(.*)', raw_content, flags=re.DOTALL)
            if think_match:
                content = think_match.group(1).strip()
            
            # If still empty, look for any text that doesn't start with <think>
            if not content:
                lines = raw_content.split('\n')
                user_facing_lines = []
                in_think_block = False
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('<think>'):
                        in_think_block = True
                        continue
                    elif line.startswith('</think>'):
                        in_think_block = False
                        continue
                    elif not in_think_block and line and not line.startswith('<'):
                        user_facing_lines.append(line)
                
                content = '\n'.join(user_facing_lines).strip()
        
        return content

    def _extract_thinking(self, raw_content: str) -> str:
        """
        Extract thinking content from AI response.
        
        Args:
            raw_content: Raw AI response content
            
        Returns:
            Extracted thinking content
        """
        import re
        
        thinking_parts = []
        
        # Extract content within <think> tags
        think_matches = re.findall(r'<think>(.*?)</think>', raw_content, flags=re.DOTALL)
        for match in think_matches:
            thinking_parts.append(match.strip())
        
        # Extract unclosed think content
        unclosed_match = re.search(r'<think>(.*)', raw_content, flags=re.DOTALL)
        if unclosed_match and not think_matches:  # Only if no closed think tags found
            thinking_parts.append(unclosed_match.group(1).strip())
        
        # Join all thinking parts
        thinking_content = '\n\n'.join(thinking_parts).strip()
        
        return thinking_content

    def _format_response_with_thinking(self, thinking: str, response: str) -> str:
        """
        Format response to include thinking process in a user-friendly way.
        
        Args:
            thinking: AI thinking content
            response: Cleaned AI response
            
        Returns:
            Formatted response with thinking
        """
        if not thinking and not response:
            return "죄송합니다. 응답을 생성하지 못했습니다."
        
        if not thinking:
            return response if response else "응답을 처리했지만 내용이 없습니다."
        
        if not response:
            return f"""<div class="thinking-section">
🤔 **AI 생각 과정:**

*{thinking}*
</div>

⏳ 응답을 생성하는 중입니다..."""
        
        # Both thinking and response exist
        formatted = f"""<div class="thinking-section">
🤔 **AI 생각 과정:**

*{thinking}*
</div>

---

💬 **답변:**

{response}"""
        return formatted

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
        .thinking-section {
            background-color: #f8f9fa;
            border-left: 4px solid #6c757d;
            padding: 10px;
            margin: 5px 0;
            font-style: italic;
            color: #6c757d;
        }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # 🤖 Attenz AI Agent
            
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
                    bubble_full_width=False,
                    show_copy_button=True,
                    render_markdown=True
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
            show_progress="minimal"
        )
        
        send_btn.click(
            handle_submit,
            inputs=[msg, chatbot], 
            outputs=[chatbot, msg],
            show_progress="minimal"
        )
        
        clear_btn.click(
            handle_clear,
            outputs=[chatbot],
            show_progress="minimal"
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
        share=True,
        show_error=False,
        show_api=True,
        enable_monitoring=True,
        max_threads=10
    )


if __name__ == "__main__":
    main() 