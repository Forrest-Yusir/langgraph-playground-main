import os
import uuid
import json
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import redis

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from utils.visualizer import visualize_graph

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers together."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a specific city."""
    return f"The weather in {city} is sunny and 25°C."

tools = [multiply, get_weather]

llm = ChatOpenAI(
    model="qwen-plus", 
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0
)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

class SimpleRedisCheckpoint:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=False)
        self.prefix = "langgraph:checkpoint"
        
    def get(self, thread_id: str) -> dict | None:
        key = f"{self.prefix}:{thread_id}"
        data = self.redis_client.get(key)
        if data:
            return json.loads(data)
        return None
    
    def save(self, thread_id: str, state: dict):
        key = f"{self.prefix}:{thread_id}"
        self.redis_client.set(key, json.dumps(state, ensure_ascii=False).encode('utf-8'))
        
    def close(self):
        self.redis_client.close()

def main():
    print("🔴 使用基础 Redis（无需 RediSearch 模块）")
    print("=" * 50)
    
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    try:
        test_client = redis.Redis.from_url(redis_url)
        test_client.ping()
        print(f"✅ Redis 连接成功：{redis_url}")
        test_client.close()
    except Exception as e:
        print(f"❌ Redis 连接失败：{e}")
        print("💡 请确保 Docker 容器已启动")
        return
    
    checkpoint_manager = SimpleRedisCheckpoint(redis_url)
    
    builder = StateGraph(AgentState)
    
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "agent")

    graph = builder.compile()

    visualize_graph(graph, "images/04_graph_structure_redis_simple.png")

    def invoke_with_checkpoint(user_input: HumanMessage, thread_id: str):
        saved_state = checkpoint_manager.get(thread_id)
        
        if saved_state:
            messages = []
            for msg in saved_state.get('messages', []):
                if msg['type'] == 'human':
                    messages.append(HumanMessage(content=msg['content']))
                else:
                    messages.append(AIMessage(content=msg['content']))
            messages.append(user_input)
            initial_state = {"messages": messages}
        else:
            initial_state = {"messages": [user_input]}
        
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke(initial_state, config=config)
        
        checkpoint_manager.save(thread_id, {
            "messages": [
                {"type": msg.type, "content": msg.content} 
                for msg in result["messages"]
            ]
        })
        
        return result
    
    config_neo = {"configurable": {"thread_id": f"user_neo_{uuid.uuid4()}"}}
    thread_id_neo = config_neo['configurable']['thread_id']
    
    print(f"\n=== Session: user_neo ===")
    print(f"Thread ID: {thread_id_neo}")
    print("User: Hi, I'm Neo.")
    result_neo = invoke_with_checkpoint(
        HumanMessage(content="Hi, I'm Neo."), 
        thread_id_neo
    )
    print(f"Agent: {result_neo['messages'][-1].content}")

    print("\nUser: What is my name?")
    result_neo = invoke_with_checkpoint(
        HumanMessage(content="What is my name?"), 
        thread_id_neo
    )
    print(f"Agent: {result_neo['messages'][-1].content}")

    config_alice = {"configurable": {"thread_id": f"user_alice_{uuid.uuid4()}"}}
    thread_id_alice = config_alice['configurable']['thread_id']
    
    print(f"\n=== Session: user_alice ===")
    print(f"Thread ID: {thread_id_alice}")
    print("User: Hi, I'm Alice.")
    result_alice = invoke_with_checkpoint(
        HumanMessage(content="Hi, I'm Alice."), 
        thread_id_alice
    )
    print(f"Agent: {result_alice['messages'][-1].content}")

    print("\nUser: What is my name?")
    result_alice = invoke_with_checkpoint(
        HumanMessage(content="What is my name?"), 
        thread_id_alice
    )
    print(f"Agent: {result_alice['messages'][-1].content}")

    print(f"\n=== Session: user_neo (Resume) ===")
    print("User neo: how are you?")
    result_neo = invoke_with_checkpoint(
        HumanMessage(content="how are you?"), 
        thread_id_neo
    )
    print(f"Agent: {result_neo['messages'][-1].content}")
    
    print("\n" + "=" * 50)
    print("✅ All sessions completed successfully!")
    print("💡 Redis 已保存所有对话状态")
    
    checkpoint_manager.close()
    
    print("\n📊 Redis 中存储的会话:")
    redis_client = redis.Redis.from_url(redis_url)
    keys = redis_client.keys("langgraph:checkpoint:*")
    for key in keys:
        print(f"  - {key.decode('utf-8')}")
    redis_client.close()


if __name__ == "__main__":
    main()