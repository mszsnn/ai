# 将之前写的 工具并发和边界处理-tool.py 的代码进行改造
# 主要是体会工业级携带状态机持久化， 防止死循环， 全自动并发的 LangGraph
import time
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from rich import print as p
from dotenv import load_dotenv


from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


load_dotenv()

# ==========================================
# 原始代码保留，不动
# ==========================================
class SQLQueryArgs(BaseModel):
    sql: str = Field(..., description="必须是只读的 SELECT 语句，严禁使用 DROP/DELETE。")


class UserStatusArgs(BaseModel):
    user_id: int = Field(..., description="用户的纯数字 ID，例如 101")


@tool(args_schema=SQLQueryArgs)
def exec_sql(sql: str) -> str:
    """执行 sql 查询"""
    p(f"[工具节点] 正在执行 SQL: [bold green]{sql}[/bold green]")
    sql_upper = sql.upper()
    if "COUNT" in sql_upper:
        return "查询成功！total_logs: 50000"
    if "LIMIT" in sql_upper:
        return "log_id: 1, action: login\n" * 10

        # 模拟危险全表扫描被拦截
    return "[FATAL ERROR: 数据量过大已截断！请立刻修改 SQL，使用 LIMIT 10 或 COUNT() 重新查询！]"


@tool(args_schema=UserStatusArgs)
def get_user_status(user_id: int) -> str:
    """查询单个用户的状态"""
    p(f"[工具节点] 网络请求查询用户 {user_id} ...")
    time.sleep(1)  # 模拟网络延迟
    return f"用户 {user_id} 状态：活跃"


tools_list = [exec_sql, get_user_status]

llm = init_chat_model(
    model='qwen/qwen3.7-flash',
    model_provider='openrouter',
    temperature=0
)

llm_with_tools = llm.bind_tools(tools_list)


# langGraph 核心定义
class AgentState(TypedDict):

    # 定义消息历史，Annotated 是python 的类型系统， 意思是： 列表类型， add_messages 信息意思是： 有新消息的时候， 不要覆盖，要新增
    messages: Annotated[list, add_messages]

# 节点1 ，负责经过大模型思考
def thought_nodes(state: AgentState):
    p('大脑思考中：\n')

    # 将message喂给模型
    result = llm_with_tools.invoke(state['messages'])

    # 不管内容是啥， 直接扔到 messages 中
    return {'messages': [result]}


# 节点2 干活的工具

action_tools = ToolNode(tools_list)

workflow = StateGraph(AgentState)

workflow.add_node('agent_brain', thought_nodes)
workflow.add_node('action_tools', action_tools)

workflow.add_edge(START, 'agent_brain')

# 添加条件边
# tools_condition 官方定义的条件判断，会拿出来最后一条，如果是工具调佣， 那就走向工具节点，如果不是工具，走向 END
workflow.add_conditional_edges('agent_brain', tools_condition, {
    'tools': 'action_tools',
    END: END
})

# 工具执行完毕之后，需要再次流回大脑
workflow.add_edge('action_tools', 'agent_brain')

app = workflow.compile()

# 初始化状态
inputs = {
    "messages": [
        SystemMessage(content="你是高级数据架构师。请高效率并发完成任务，遇到数据截断要主动修改查询策略。"),
        HumanMessage(content="帮我把数据库里全部日志查出来。同时核实一下用户 101, 102, 103 的状态。")
    ]
}

final_state = app.invoke(inputs)


# 2. 直接从返回的最终状态字典里提取最后一条消息
p("\n🤖 [bold magenta]最终完美回答：[/bold magenta]")
p(final_state["messages"][-1].content)


