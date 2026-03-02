
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from utils.visualizer import visualize_graph

load_dotenv()

# 1. State
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. Tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers together."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a specific city."""
    return f"The weather in {city} is sunny and 25°C."

tools = [multiply, get_weather]

# 3. Nodes
llm = ChatOpenAI(
    model="qwen-plus", 
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0
    )
# llm = ChatQwen(
#     model="qwen-plus", 
#     temperature=0
#     )
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    print("--- [Node: Agent] Thinking... ---")
    result = llm_with_tools.invoke(state["messages"])
    print(f"--- [Node: Agent] Output: {result.content} (Tool Calls: {len(result.tool_calls)})")
    return {"messages": [result]}

# 4. Graph Construction
def main():
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    # 使用自定义的 ToolNode 以便于打印日志 (可选，如果不介意直接用 prebuilt)
    # 但为了简单，我们还是用 prebuilt ToolNode，它本身不打印日志。
    # 我们可以通过监听 graph 的输出来实现更直观的流式展示
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    
    # [高级工程师写法]: 
    # 使用 tools_condition (标准) + 显式 Path Map (清晰)
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )

    builder.add_edge("tools", "agent")

    graph = builder.compile()

    # Visualization
    visualize_graph(graph, "03_graph_structure.png")

    # Run
    print("\n=== Test: Weather Query ===")
    initial_input = {"messages": [HumanMessage(content="What is the weather in Boston?")]}
    print(f"Initial Input: {initial_input['messages'][0].content}")
    
    print("\n--- Start Streaming ---")
    # 使用 stream 模式来实时查看每一步的输出
    # stream_mode="updates" 会返回每个节点更新后的状态增量
    for event in graph.stream(initial_input, stream_mode="updates"):
        for node_name, state_update in event.items():
            print(f"\n[Update from Node: {node_name}]")
            
            # 解析消息更新
            if "messages" in state_update:
                messages = state_update["messages"]
                for msg in messages:
                    # 根据消息类型打印不同格式
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                         # AI Message 且包含工具调用
                        for tc in msg.tool_calls:
                            print(f"  🤖 AI Request Tool: {tc['name']} (Args: {tc['args']})")
                    elif hasattr(msg, "content") and msg.content:
                        # 普通内容
                        prefix = "  🛠️ Tool Output" if node_name == "tools" else "  🤖 AI Message"
                        print(f"{prefix}: {msg.content}")
                    else:
                        print(f"  (Raw Message): {msg}")
    
    print("\n--- End Streaming ---")

if __name__ == "__main__":
    main()