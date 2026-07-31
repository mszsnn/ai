# 在流程执行的时候，一些高危的操作， 需要人的参与

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class AgentState(TypedDict):
    user_input: str
    generated_sql: str
    execution_result: str


def llm_reasoning_node(state: AgentState):
    # 模拟ai 思考，将自然语言，转化为 SQl
    print('ai 思考中')
    # 假设大模型生成了一句高危 SQL
    return {"generated_sql": "DELETE FROM business_logs WHERE date < '2023-01-01'"}


def execution_sql_node(state: AgentState):
    print(f"🔥 [数据库执行] 正在向数据库发送: {state['generated_sql']}")
    # 模拟执行成功
    return {"execution_result": "成功删除 1542 条记录。"}

workflow = StateGraph(AgentState)

workflow.add_node('llm_reasoning_node', llm_reasoning_node)

workflow.add_node('execution_sql_node', execution_sql_node)

workflow.add_edge(START, 'llm_reasoning_node')
workflow.add_edge('llm_reasoning_node', 'execution_sql_node')
workflow.add_edge('execution_sql_node', END)

memory  = MemorySaver()

app = workflow.compile(
    checkpointer=memory,
    interrupt_before=['execution_sql_node']
)

initial_input = {"user_input": "帮我清理一下去年的旧日志数据。"}

# 设定一个对话线程 ID (代表当前用户的这次会话)
thread_config = {"configurable": {"thread_id": "chat_001"}}
for event in app.stream(initial_input, config=thread_config):
    pass

current_state = app.get_state(thread_config)

print("流程已挂起！下一步准备执行的节点是:", current_state.next)
print("请将这句 SQL 推送给前端用户审批:", current_state.values['generated_sql'])


# 接下来需要用户介入
user_approval = input('管理员，是否允许执行上述 SQL? y/n')

if user_approval.lower() == 'y':
    print('继续执行 sql ')
    for event in app.stream(None, config=thread_config):
        pass
    final_state = app.get_state(thread_config)
    print('最终结果', final_state.values['execution_result'])
else:
    print('流程终止')



