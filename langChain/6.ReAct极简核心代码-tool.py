from langchain_core.messages import  HumanMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from dotenv import load_dotenv
from rich import print as p
load_dotenv()
# 制造工具
@tool
def get_user_email(name:str) -> str:
    """ 根据姓名查询企业邮箱地址"""
    # 这里模拟真实的数据库查表操作
    mock = {"张三": "zhangsan@company.com", "李四": "lisi@company.com"}
    return mock.get(name, '数据库报错，查无此人，请检查用户名')


# 初始化大模型， 让大模型识别工具

llm = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider='openrouter',
    temperature=0,
)


# 绑定工具
llm_with_tool = llm.bind_tools([get_user_email])


# 构造智能体的记忆
message = [
    HumanMessage(content='帮我查一下李四的邮箱是多少')
]


p('启动纯手工 agent 大循环 \n')

# ==========================================
# 主战场：手工 While/For 循环
# ==========================================

for step in range(5):
    p(f"---[step{step + 1}]-模型大脑推理中----")

    # 发送消息
    response = llm_with_tool.invoke(message)

    message.append(response)

    # 没有调用工具， 直接输出结果
    if not response.tool_calls:
        p('\n最终回答', response.content)
        break

    # 调用工具

    for tool_call in response.tool_calls:
        p(f'模型触发工具调用，工具名称：{tool_call["name"]} 参数：{tool_call["args"]}  id: {tool_call["id"]}\n')

        if tool_call['name'] == 'get_user_email':
            # 调用函数
            result = get_user_email.invoke(tool_call['args'])
            p(f'本地工具执行结果{result}\n')

            # 本地的结果要生成工具的调用结果，传递给ai

            tool_mes = ToolMessage(
                content=str(result),
                tool_call_id=tool_call['id']
            )

            message.append(tool_mes)
            p("♻️ 已将工具结果压入记忆，准备进入下一轮循环让模型反思...\n")



