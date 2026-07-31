from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import  HumanMessage, AIMessage, ToolMessage
from rich import print as p


# 定义核心大脑： 生产级 Prompt 模版
prompt = ChatPromptTemplate.from_messages([
     ('system', '你是一个高级数据库智能体, 请严格基于以下企业数据字典回答问题\n{context}'),
     MessagesPlaceholder(variable_name='chat_history'),
     ('human', '{user_input}'),
     MessagesPlaceholder(variable_name='agent_scratchpad')
])

# ==========================================
# 2. 模拟真实运行时的动态数据流
# ==========================================

# A 模拟： 从向量数据库检索回来的私有知识（rag)
retrieved_context = '数据字典】表名: users | 字段: id (主键), user_name (姓名), age (年龄)'

# B 模拟： 用户之前的对话记录

history = [
    HumanMessage('你们的数据库里面到底存储了什么？'),
    AIMessage(content='我们主要存储了企业内部用户的基础信息')
]

# (C) 模拟：智能体“草稿本” (ReAct 纠错核心)
# 假设大模型刚刚生成了一句错误的 SQL，工具执行报错了。我们需要把这个过程塞回去，让它反思。
scratchpad = [
    # 模型上一步的输出：决定调用 execute_sql 工具，但把表名写错了 (写成了 user)
    AIMessage(
        content="",
        tool_calls=[{
            "name": "execute_sql",
            "args": {"sql_query": "SELECT user_name FROM user;"},
            "id": "call_abc123"
        }]
    ),
    # 你的本地 Python 函数捕获到的报错信息，封装成 ToolMessage 还给模型
    ToolMessage(
        tool_call_id="call_abc123",
        content="Execution Error: Table 'user' doesn't exist. Did you mean 'users'?"
    )
]

final_messages = prompt.format_messages(
    context=retrieved_context,
    chat_history=history,
    user_input="帮我查一下所有用户的名字。",
    agent_scratchpad=scratchpad
)


print("==== 最终发给大模型的 Messages 数组 ====\n")
p(final_messages)