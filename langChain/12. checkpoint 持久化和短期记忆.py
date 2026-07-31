from http.client import responses
from typing import Annotated

from spacy.lang.en.tokenizer_exceptions import word
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from rich import print as p
from dotenv import  load_dotenv

load_dotenv()


class AgentState(TypedDict):
    # 全部的会话记录
    messages: Annotated[list, add_messages]

    # summary 专门开放一块区域，用来存放高度压缩的记忆
    summary: str


llm = init_chat_model(
    model='moonshotai/kimi-k3-free',
    model_provider='openai',
    temperature=0
)

 # 节点 A， 主干， 负责根据现有记忆回答

def call_model(state: AgentState):
    p('[主节点] 正在调用大模型思考\n')

    # 如果有摘要信息， 那就需要将摘要也挂给当前的信息
    summary = state.get('summary', '')

    if summary:
        sys_message = f'以下是你和用户历史聊天摘要， 请牢记： \n{summary}'
        messages = [HumanMessage(content=sys_message)] + state['messages']
    else:
        messages = state['messages']

    response = llm.invoke(messages)

    return {'messages': [response]}


# 节点 B 垃圾回收和记忆压缩

def summarize_and_trim(state: AgentState):
    p("[内存回收节点触发] 消息过长！正在压缩记忆并清理大巴车...\n")
    summary = state.get('summary', '')
    messages = state['messages']
    # 只保留最新的两条， 其余全部压缩
    messages_to_compress = messages[:-2]
    new_history = '\n'.join([f'{ m.type }: {m.content}' for m in messages_to_compress])

    # 压缩消息，生成新的摘要
    summary_prompt = (
        f"请把下面的聊天记录，融合到现有的聊天摘要中\n"
        f"现有的摘要： {summary} \n"
        f'新聊天记录：\n {new_history}'
    )
    new_summary = llm.invoke([
        HumanMessage(content=summary_prompt)
    ])

    p(f"[生成记忆胶囊] 新摘要: {new_summary} \n")

    # 将原来的删除
    remove_message = [RemoveMessage(id = m.id) for m in messages_to_compress]

    return {
        'summary': new_summary,
        'messages': remove_message
    }

def check_condition(state: AgentState):
    """ 检查什么时候应该去清理历史记录 """
    messages = state.get('messages', [])

    if len(messages) > 6:
        return 'summarize_and_trim'
    return END


workflow = StateGraph(AgentState)

workflow.add_node('call_model', call_model)
workflow.add_node('summarize_and_trim', summarize_and_trim)

workflow.add_edge(START, 'call_model')
workflow.add_conditional_edges('call_model', check_condition)
workflow.add_edge('summarize_and_trim', END)

# 进行存储
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)



### --------------###


config = {"configurable": {"thread_id": "user_001"}}

# 灌入一些数据
conversations = [
    "你好，我叫张三，我的工号是 9527。请介绍下你自己， 你是 Kimi 的哪个版本？",
    "我最喜欢的颜色是蓝色。",
    "我中午打算去吃兰州拉面。"
]

for user_input in conversations:
    p(f"\n 用户: {user_input}")
    result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
    p(f" AI: {result['messages'][-1].content}")

# 查看当前状态（此时应该有 6 条消息：3个User + 3个AI）
current_state = app.get_state(config)
p(f"\n [监控] 当前消息总数：{len(current_state.values['messages'])}")


# 再发一条，强行突破 6 条的阈值！
p("\n 用户: 你还记得我第一句说了什么吗？")
result = app.invoke({"messages": [HumanMessage(content="你还记得我第一句说了什么吗？")]}, config)
p(f"AI: {result['messages'][-1].content}")


# 看看内存回收后，大巴车上还剩几条？
final_state = app.get_state(config)
p(f"[监控] 压缩后的消息总数：{len(final_state.values['messages'])}")
p(f"[监控] 永久保存的摘要：{final_state.values['summary']}")





