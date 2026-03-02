# LangGraph Playground

这是一个用来学习 [LangGraph](https://langchain-ai.github.io/langgraph/) 的项目。

我会在这里使用 LangGraph 编写一些 AI Agent 示例和项目，用于探索和实践 Agentic Workflow。

## 📚 Tutorials

这里是一系列循序渐进的教程，帮助你理解 LangGraph 的核心概念：

- **[01_state_and_nodes.py](tutorials/01_state_and_nodes.py)**
  - 基础入门：介绍 `StateGraph` 的构建。
  - 核心概念：`State` 定义 (TypedDict)、简单节点 (Nodes) 的编写、线性图结构。
  
- **[02_edges_and_routing.py](tutorials/02_edges_and_routing.py)**
  - 路由控制：介绍 `Conditional Edges` (条件边)。
  - 核心概念：Router 逻辑编写、根据 State 动态决定下一步走向 (分支逻辑)。

- **[03_tool_calling.py](tutorials/03_tool_calling.py)**
  - 工具调用：结合 LLM 进行 Tool Calling。
  - 核心概念：`bind_tools`、`ToolNode`、`tools_condition` 以及如何流式输出 (Streaming) 运行状态。

- **[04_persistence.py](tutorials/04_persistence.py)**
  - 记忆持久化：让 Agent 拥有"记忆"。
  - 核心概念：Checkpointer (`SqliteSaver`)、`thread_id` 会话管理、跨请求的状态恢复与隔离。
  - 个人笔记：为了实现Time_travel,要确保每次superstep结束时，通过Checkpointer来自动保存完整的状态，即 状态快照。要注意在agent工作流compile编译阶段时传入checkpointer，然后运行时自动处理所有持久化操作。其中持久化的是利用python内置的sqlite配合SqliteSaver来实现的。
  
- **[05_human_in_the_loop.py](tutorials/05_human_in_the_loop.py)**
  - 人机交互 (HITL)：在 Agent 执行过程中加入人工干预。
  - 核心概念：`interrupt_before` 断点机制、人工审批/拒绝/修改工具调用、图的暂停与恢复 (Resuming)。
  - 个人笔记：这里的全局状态只有一个 messages 字段存储完整的对话历史-chat模式。
    - **[Chat Mode]：（第8个案例demo）

- **[06_parallelism_map_reduce.py](tutorials/06_parallelism_map_reduce.py)**
  - 并行处理：Map-Reduce 模式。
  - 核心概念：`Send` API 实现动态并行分支 (Map)、`operator.add` 聚合器 (Reduce)、并发状态管理。

- **[07_hybrid_subgraphs.py](tutorials/07_hybrid_subgraphs.py)**
  - 混合架构：父子图 (Subgraphs) 嵌套。
  - 核心概念：将不同架构（如 ReAct 和 Map-Reduce）封装为独立子图、层级状态管理、复杂工作流的模块化复用。

- **[08_multi_agent_supervisor](tutorials/)**
  - 多智能体协作：Supervisor (主管) 模式。
  - 核心概念：中心化路由控制、结构化输出做决策。
  - 变体：
    - **[Chat Mode](tutorials/08_multi_agent_supervisor_chat.py)**: 基于对话历史的协作。
        - class AgentState(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]  # 单一聊天记录
            next: str
        - 特点：只有一个 messages 字段存储完整的对话历史
        - 工作方式：像传统聊天机器人一样，所有信息都在一个消息列表中传递
        - 缺点：状态分散在消息中，难以精确控制和追踪
        - 设计哲学：Chat模式体现了"对话即状态"的思想 - 把整个协作过程当作一场对话来处理。NLP友好的多智能
    - **[Artifact Mode](tutorials/08_multi_agent_supervisor_artifact.py)**: 围绕特定工件 (Artifact) 的迭代优化。
      -  class AgentState(TypedDict):
            request: str           # 原始需求 (Single Source of Truth)
            code: str             # 当前代码版本
            review: str           # 评审意见
            revision_number: int  # 迭代计数
            next: str             # 控制流
            messages: ...         # 审计日志(可选)
        - 特点：每个重要信息都有独立的字段管理
        - 工作方式：像工单系统一样，每个状态都有明确的所有者
        - 优点：状态清晰，易于调试和监控
        - 设计哲学：Artifact模式体现了"工单驱动"的思想 - 把协作过程结构化为明确的工作项流转。 工程化的多智能体
- **[09_multi_agent_handoff.py](tutorials/09_multi_agent_handoff.py)**
  - 多智能体接力：Handoff (Swarm) 模式。
  - 核心概念：`Command` API 实现命令式跳转、去中心化控制、Agent 之间显式交接棒 (Context Passing)。

- **[10_plan_and_execute.py](tutorials/10_plan_and_execute.py)**
  - 规划与执行：Plan-and-Execute 模式。
  - 核心概念：Planner (规划)、Executor (执行)、Re-Planner (反思与动态调整) 的闭环循环，处理长链路复杂任务。
