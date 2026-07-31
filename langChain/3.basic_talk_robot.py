from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from rich import print as p


load_dotenv()


llm = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider='openrouter',
    temperature= 0,
)

# 基础的一个保存会话历史的聊天机器人

# 维护会话的列表

session_history = [SystemMessage('你是一个万能助手')]

EXIT_INPUT = 'quit'
p(f"聊天会话开始,请输入具体的问题， 当输入{EXIT_INPUT} 的时候，会话结束", '\n')

n = 1
while True:

    p(f'=====当前对话是第{n}轮开始--------', '\n')

    user_input = input('请输入：\n')

    if user_input == EXIT_INPUT:
        p('会话已经结束，欢迎下次再来', '\n')
        break

    p(f'用户提问：{user_input}', '\n')
    session_history.append(HumanMessage(content=user_input))

    resp = llm.invoke(session_history)
    p(f'小助手回答：{resp.content}', '\n')
    session_history.append(AIMessage(content=resp.content))

    p(f'=====当前对话是第{n}轮结束--------', '\n')
    n = n + 1


