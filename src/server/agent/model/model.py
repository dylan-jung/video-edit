import os
from typing import Any, Dict, List

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage, ToolMessage)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.server.agent.model.state import AgentState
from src.server.agent.prompt import (executer_prompt, front_man_prompt,
                                     planner_prompt, rewrite_prompt)
from src.server.agent.tools import ToolExecutor
from src.server.utils.cookie_utils import get_code_server_cookies


def initialize_llm():
    """Initialize the language model with proper configuration."""
    return ChatOpenAI(
        model=os.getenv('AGENT_MODEL', 'gpt-4o'),
        openai_api_base=os.getenv('AGENT_MODEL_SERVER'),
        openai_api_key=os.getenv('AGENT_MODEL_API_KEY', "EMPTY"),
        streaming=True,
        extra_headers={
            "Cookie": f"code-server-session={get_code_server_cookies().get('code-server-session')}"
        }
    )


def create_agent_workflow(state_type):
    """Create the agent workflow with proper nodes and edges."""
    llm = initialize_llm()
    tool_executor = ToolExecutor()
    tools = tool_executor.as_tool()
    llm_with_tools = llm.bind_tools(tools)
    
    workflow = StateGraph(state_type)
    
    def rewrite_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        response = llm.invoke(messages + [SystemMessage(content=rewrite_prompt)])
        content = response.content.strip()
        return {"messages": messages + [AIMessage(content=content)]}
    
    def route_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        response = llm.invoke(messages + [SystemMessage(content="당신은 현재의 콘텍스트에서 사용자의 요청을 수행할 수 있는지 판단해야합니다. 답변할 수 있다면 <yes> 그게 아니라면 <no>를 출력해주세요. 만약 요청을 수행할 수 없다면 그 이유를 설명해주세요.")])
        content = response.content
        print(f"🤖 Route Response: {content}")
        if "<yes>" in content:
            return {"messages": messages, "route": "front_man_state"}
        elif "<no>" in content:
            return {"messages": messages, "route": "plan_state"}
        raise ValueError(f"Invalid route response: {content}")

    def route_simple(state: AgentState) -> AgentState:
        route = state.get("route", END)
        return route

    def planner_node(state: AgentState) -> AgentState:
        """Analyzes user intent and plans the execution strategy."""
        messages = state["messages"]
        
        # Create a system message for intent analysis
        planning_llm = llm_with_tools
        
        # Add system message for planning
        planning_messages = [SystemMessage(content=planner_prompt(state["project_id"], state["timestamp"]))]
        for message in messages:
            if isinstance(message, (AIMessage, HumanMessage)):
                planning_messages.append(message)
        
        # Get the analysis and plan
        plan_response = planning_llm.invoke(planning_messages)
        content = plan_response.content.strip()
        return {
            "messages": messages,
            "plan": content
        }

    def executer_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        plan = state.get("plan", "")
        plan_messages = state.get("plan_messages", [])
        if len(plan_messages) == 0:
            plan_messages = [SystemMessage(content=executer_prompt(state["project_id"], state["timestamp"])), HumanMessage(content=plan)]
        response = llm_with_tools.invoke(plan_messages)
        content = response.content.strip()
        plan_messages_with_response = plan_messages + [response]
        print(f"🤖 Executer Response: {content}")
        if "<done>" in content.strip().lower():
            results = []
            for msg in plan_messages:
                if isinstance(msg, ToolMessage) or isinstance(msg, AIMessage):
                    results.append(msg)
            return {"messages": messages + results, "route": "route_state"}

        if hasattr(response, "tool_calls") or response.tool_calls:
            return {"messages": messages, "plan": plan, "plan_messages": plan_messages_with_response, "route": "tools_state"}
        else:
            return {"messages": messages, "plan": plan, "plan_messages": plan_messages_with_response, "route": "execute_state"}
    
    def tools_node(state: AgentState) -> AgentState:
        """Execute tools based on the tool calls in the last message."""

        messages = state["messages"]
        plan = state.get("plan", "")
        plan_messages = state.get("plan_messages", [])

        last_message = plan_messages[-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {"messages": messages, "plan": plan, "plan_messages": plan_messages}
        
        tool_responses: List[ToolMessage] = []
        for tool_call in last_message.tool_calls:
            # Extract tool call data
            tool_name = tool_call['name']
            tool_args = tool_call['args'] if 'args' in tool_call else {}
            tool_id = tool_call['id'] if 'id' in tool_call else ""
            # Format for the tool executor
            formatted_tool_call = {
                "name": tool_name,
                "arguments": tool_args
            }
            print(f"🔧 Tool Call: {tool_name}")
            # Execute the tool
            result = tool_executor.execute_tool(formatted_tool_call)
            
            # Create a tool message with the result
            tool_message = ToolMessage(
                content=result,
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_responses.append(tool_message)
        
        return {"messages": messages, "plan": plan, "plan_messages": plan_messages + tool_responses}
    
    def front_man_node(state: AgentState) -> AgentState:
        messages = state.get("messages", [])
        response = llm.invoke(messages + [SystemMessage(content=front_man_prompt)])
        content = response.content.strip()
        return {"messages": messages + [AIMessage(content=content)]}
    
    # Add nodes to the graph
    # 1. Collecting Information
    workflow.add_node("rewrite_state", rewrite_node)
    workflow.add_node("route_state", route_node)
    workflow.add_node("plan_state", planner_node)
    workflow.add_node("execute_state", executer_node)
    workflow.add_node("tools_state", tools_node)

    # 2. Describe the result
    workflow.add_node("front_man_state", front_man_node)
    
    # Add edges to the graph
    # 1. Collecting Information
    workflow.add_edge(START, "rewrite_state")
    workflow.add_edge("rewrite_state", "route_state")
    workflow.add_conditional_edges("route_state", route_simple, {
        "plan_state": "plan_state",
        "front_man_state": "front_man_state",
        END: END
    })
    workflow.add_edge("plan_state", "execute_state")
    workflow.add_conditional_edges(
        "execute_state", 
        route_simple,
        {
            "tools_state": "tools_state",
            "execute_state": "execute_state",
            "route_state": "route_state"
        }
    )
    workflow.add_edge("tools_state", "execute_state")

    # 2. Describe the result
    workflow.add_edge("front_man_state", END)
    
    # Compile the workflow before returning
    return workflow.compile()
