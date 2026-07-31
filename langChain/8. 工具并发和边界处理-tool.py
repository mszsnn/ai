# 在真实的工业场景中
# 工具的使用必须是严谨的
# 1. 决不能通过注释来来约束工具的入参二号出参数
# 2. 必须有执行并发能力，或者使用异步
# 3. 不能无条件相信工具的执行结果，我们要进行一定的边界处理




####——————————————————————————————####

# 这个例子表现出不同的模型的思考特性和能力特征
# * 有的模型压根不能识别调用工具
# * 有的模型可以识别并行， 有的模型识别不了并行
# * 有的模型有更深层次的思考，宏观的考虑，加小步的探索


import time
import concurrent.futures
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from rich import print as p
from dotenv import load_dotenv

load_dotenv()
# 用pydantic 来将工具的定义强约束， 可以添加描述、 参数的校验、生成 json schema

class SQLQueryArgs(BaseModel):
    # ... 代表必填， 不能为空
    sql: str = Field(..., description='必须是只读的 SELECT 语句，严禁使用 DROP/DELETE 等危险操作。')

class UserStatusArgs(BaseModel):
    user_id: int = Field(..., description="用户的纯数字 ID，例如 101")

@tool(args_schema=SQLQueryArgs)
def exec_sql(sql: str) -> str:
    """执行 sql 查询"""
    p(f'正在执行 sql 查询: {sql} \n')  # 打印出模型真实的SQL，方便我们观察

    sql_upper = sql.upper()

    # 💡 增加逻辑分支：如果模型听了劝，用了 COUNT 或 LIMIT，就给它正确的返回！
    if "COUNT" in sql_upper:
        return "查询成功！total_logs: 50000"

    if "LIMIT" in sql_upper:
        # 只返回10条，不触发截断
        return "log_id: 1, action: login\n" * 10

        # 如果模型头铁，非要查全部，才触发灾难警告
    raw_huge_data = "log_id: 1, action: login\n" * 50000
    MAX_LEN = 1000

    if len(raw_huge_data) > MAX_LEN:
        trunc_data = raw_huge_data[:MAX_LEN]
        warning_msg = (
            f"\n...[FATAL ERROR: 返回数据高达 {len(raw_huge_data)} 字符，已在此处强制截断！"
            f"直接读取这么多数据会撑爆你的显存。请反思：立刻修改你的 SQL，使用 LIMIT 10 或 COUNT() 聚合函数重新查询！]..."
        )
        return trunc_data + warning_msg

    return raw_huge_data


@tool(args_schema=UserStatusArgs)
def get_user_status(user_id: int) -> str:
    """查询单个用户的状态"""
    p(f"[API 请求] 正在网络请求查询用户 {user_id} 的状态...")
    time.sleep(2) # 模拟网络延迟 2 秒
    return f"用户 {user_id} 状态：活跃"

# 映射字典
tools_map = {
    "exec_sql": exec_sql,
    "get_user_status": get_user_status
}

## agent

llm = init_chat_model(
    # model="openai/gpt-oss-120b",
    model_provider='openrouter',
    # model='qwen/qwen3.7-flash',
    # model='minimax/minimax-m3',
    model='xiaomi/mimo-v2.5-pro',
    temperature=0,
    max_tokens=8000
)

llm_with_tools = llm.bind_tools([
    exec_sql,
    get_user_status
])



messages = [
    SystemMessage(content='你是高级数据架构师。请高效率地完成用户的任务，遇到数据截断要主动修改查询策略。'),
    HumanMessage(content='帮我把数据库里全部的日志查出来。同时，帮我分别核实一下用户 101, 102, 103 的状态。')
]

p('启动执行 \n')

for step in range(5):
    p(f"\n--- [第 {step + 1} 步] 模型大脑思考中 ---")
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        p("\n 最终回答:\n", response.content)
        break

    # 并发执行

    p(f"模型一次性下达了 {len(response.tool_calls)} 个独立任务！")

    # 定义一个单任务执行器

    def execute_single_tool(tool_call):
        func_name = tool_call['name']
        func_args = tool_call['args']
        tool_id = tool_call['id']
        p(f'任务名称：{func_name}, 参数：{func_args} \n')
        func = tools_map[func_name]
        try:
            res = func.invoke(func_args)
            return ToolMessage(
                content=str(res),
                tool_call_id=tool_id
            )
        except Exception as e:
            return ToolMessage(
                content=f'执行异常 Error: {e} ',
                tool_call_id=tool_id
            )


    # 开启并发执行
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tool_results = list(executor.map(execute_single_tool, response.tool_calls))

    p(f"⏱️ 并发执行完成！耗时: {time.time() - start_time:.2f} 秒")

    messages.extend(tool_results)
    p("♻️ 已将多份执行结果喂给大模型，准备进入下一轮反思...")














