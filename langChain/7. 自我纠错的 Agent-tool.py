from langchain_core.messages import  HumanMessage, ToolMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from dotenv import load_dotenv
from rich import print as p
load_dotenv()

# 制造两个工具
@tool
def get_id(name: str) -> str:
    """ 查询员工数据库 查询对应的数字工号"""
    p(f'正在查询员工{name}的工号 \n')
    db = {"李四": "1024", "王五": "2048"}
    return db.get(name, '查不到数据')

@tool

def get_salary(id: str) -> str:
    """ 查询员工的薪水，注意入参必须是纯数字工号"""
    p(f'正在查询员工ID{id}的薪水 \n')

    if not id.isdigit():
        # 异常报错
        raise ValueError(f'error: id 字段必须是纯数字')
    return '10000.00 RMB ' if id == '1024' else '0 RMB'

llm = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider='openrouter',
    temperature=0,
)

llm_with_tools = llm.bind_tools([
    get_id,
    get_salary
])

messages = [
    SystemMessage(content="你是一个严谨的HR助手。如果遇到报错，请不要向用户抱怨，而是自己想办法利用其他工具解决问题！获取结果后立刻用中文回答。"),
    HumanMessage(content="帮我查一下李四每个月多少钱？")
]

print("--启动的 Agent 大循环...\n")

tools_map = {
    'get_id': get_id,
    'get_salary': get_salary
}

for step in range(5):
    print(f"--- [第 {step + 1} 步] 模型大脑思考中 ---")
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # 如果没调工具，说明大模型觉得完事了
    if not response.tool_calls:
        print("最终回答:", response.content)
        break

    for tool_call in response.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_id = tool_call['id']

        print(f" 模型决定调用: [{tool_name}]，参数: {tool_args}] \n")

        exec_tool = tools_map[tool_name]

        try:
            exec_result = exec_tool.invoke(tool_args)
            result = str(exec_result)
            p(f"执行成功: {result} \n")
        except Exception as e:
            # 报错
            result = f'System Exception: {str(e)}'
            print(f"❌ 执行崩溃，已捕获异常: {result} \n")


        messages.append(ToolMessage(
            content=result,
            tool_call_id=tool_id
        ))
        print("已将【执行结果/报错信息】喂给大模型，准备进入下一轮反思...\n")



