import os
import sqlite3
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages  import SystemMessage, HumanMessage, ToolMessage, AIMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver  # SQLite 持久化记忆
from langgraph.checkpoint.memory import MemorySaver  # 内存级临时记忆
from openai import RateLimitError, AuthenticationError, BadRequestError, APIConnectionError
from pathlib import Path
from library_agent.agent.tools import agent_tools

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SQLLite_PATH = PROJECT_ROOT / "checkpoints.sqlite"


# ==========================================
# 1. State ：数据
# ==========================================

class AgentState(TypedDict):
    """
    messages 增量消息数组， 使用 add_message 自动提添加或者删除消息
    summary 用来存放被压缩的历史对话背景
    """

    messages: Annotated[list, add_messages]
    summary: str

# ==========================================
# 2. 大脑
# ==========================================

class BookAgent:
    """
    核心 agent
    """
    def __init__(self):

        self.llm = init_chat_model(
            model=os.getenv("LLM_MODEL"),
            model_provider=os.getenv("LLM_PROVIDER"),
            api_key=os.getenv("LLM_OPENAI_API_KEY"),
            base_url=os.getenv("LLM_OPENAI_BASE_URL"),
            temperature=0  # 防幻觉：设为 0
        )

        self.llm_with_tools = self.llm.bind_tools(agent_tools)

    def _agent_node(self, state: AgentState):
        """
        思考节点： 挂在 System prompt 和 历史 Summary
        """
        summary = state.get('summary', '')

        # 定义系统级 prompt
        sys_prompt_text = (
            "你是一个专业的图书智能体。\n"
            "【核心行为准则】：\n"
            "1. 当用户询问关于本书的具体内容、概念、原则或知识点时，你必须且只能通过调用 `search_keyword_tool` 工具进行查阅。\n"
            "2. 严禁基于你自身的先验知识随意编造答案！如果工具返回未找到相关信息，请如实回答“根据本书内容，未查阅到相关记载”。\n"
            "3. 引用工具返回的原文回答时，请在回答末尾明确标注【来源出处】（如：来源 Page X / Lines X-Y）。"
        )

        if summary:
            sys_prompt_text = sys_prompt_text + f"\n\n【之前对话的背景摘要】：\n{summary}"

        system_prompt = SystemMessage(content=sys_prompt_text)

        full_prompt = [system_prompt] + state['messages']


        # 加上防御机制
        try:
            response = self.llm_with_tools.invoke(full_prompt)
            return {'messages':[response]}

        # 致命错误，不再重试
        except (RateLimitError, AuthenticationError, APIConnectionError) as fatal_err:
            print(f"[熔断器触发] 遭遇致命物理错误: {type(fatal_err).__name__}")
            # 伪造一条 AI 消息返回给用户，图网络会在下一步安全走向 END
            error_msg = AIMessage(
                content="[系统提示] 抱歉，AI 大脑暂时失去连接（可能由于 API 额度耗尽或网络波动）。请稍后再试或联系管理员。"
            )
            return {"messages": [error_msg]}

        # 重试， 最多一次
        except BadRequestError as bad_req_err:
            print(f"[自动自愈] 拦截到上下文错乱 (400): {str(bad_req_err)[:50]}")
            last_msg = state["messages"][-1]

            # 熔断保护：检查是不是已经重试过了？
            # 如果最后一条消息已经是我们自己发的容错提示，说明重试失败了，直接放弃！
            if isinstance(last_msg, SystemMessage) and "[系统级容错]" in last_msg.content:
                print("[熔断器触发] 自愈失败，停止死循环！")
                return {"messages": [AIMessage(
                    content="[系统提示] 对话状态严重异常，系统尝试自愈失败，请尝试开启新的会话 (清空 Thread ID)。")]}

            # 执行手术清创
            if hasattr(last_msg, "id") and last_msg.id:
                return {
                    "messages": [
                        RemoveMessage(id=last_msg.id),
                        SystemMessage(content="[系统级容错] 刚才的上下文格式错乱，脏数据已清理。请重新思考并回答用户。")
                    ]
                }
        except Exception as unknown_err:
            print(f"[未知异常] {str(unknown_err)}")
            return {"messages": [AIMessage(content=f"[系统提示] 发生未知内部错误: {str(unknown_err)}")]}


    def _summarize_and_trim(self, state: AgentState):
        print("[内存回收节点触发] 消息过长！正在压缩记忆...\n")

        summary = state.get('summary', '')

        messages = state['messages']
        # 只保留最新的1条， 其余全部压缩， 最后一条是最终输出的回答
        messages_to_compress = messages[:-1]
        new_history = '\n'.join([f'{m.type}: {m.content}' for m in messages_to_compress])

        # 压缩消息，生成新的摘要
        summary_prompt = (
            f"请把下面的聊天记录，融合到现有的聊天摘要中\n"
            f"现有的摘要： {summary} \n"
            f'新聊天记录：\n {new_history}'
        )
        new_summary = self.llm.invoke([
            HumanMessage(content=summary_prompt)
        ])

        print(f"[生成记忆胶囊] 新摘要: {new_summary} \n")

        # 将原来的删除
        remove_message = [RemoveMessage(id=m.id) for m in messages_to_compress]

        return {
            'summary': new_summary.content,
            'messages': remove_message
        }

    def _route_after_agent(self, state: AgentState):
        """
        1 如果llm 下达调用 tool， 那就优先走向 action_tools 执行工具
        2 如果无需调用工具， 检查历史消息是否挤压过多
        """
        messages = state.get('messages', [])
        last_message = messages[-1]

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return 'action_tools'

        if len(messages) > 6:
            return 'summarize_and_trim'

        return END


    def build_graph(self, use_sqlite: bool = True, db_path: str = DEFAULT_SQLLite_PATH):
        """
        拼接节点，构建执行图
        """

        workflow = StateGraph(AgentState)

        # 挂在节点
        workflow.add_node('agent_brain', self._agent_node)
        workflow.add_node('action_tools', ToolNode(agent_tools, handle_tool_errors=True))
        workflow.add_node('summarize_and_trim', self._summarize_and_trim)

        #  设置流程
        workflow.add_edge(START, 'agent_brain')
        workflow.add_conditional_edges(
            'agent_brain',
            self._route_after_agent
        )

        workflow.add_edge('action_tools', 'agent_brain')
        workflow.add_edge('summarize_and_trim', END)

        if use_sqlite:
            connet = sqlite3.connect(db_path, check_same_thread=False)
            checkpointer = SqliteSaver(connet)
            print('SQLite 长期持久化已经做好')
        else:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
        return  workflow.compile(checkpointer = checkpointer)


# ==========================================
# ReAct 测试
# ==========================================
if __name__ == "__main__":
    print("正在初始化 LangGraph 工作流...")

    builder = BookAgent()
    app = builder.build_graph(use_sqlite=True)

    test_book_id = "agile_project_management"  # 对应上一阶段存入 Chroma 的集合名

    session_config = {
        "configurable": {
            "thread_id": "session_demo_003",  # 记忆追踪卡槽
            "book_id": test_book_id  # 穿透至 search_knowledge_base 工具的租户卡槽
        }
    }

    # 3. 轮次 1：需要查书的提问
    user_query_1 = "敏捷开发的核心价值观是什么？请给出具体来源。"
    print(f"[User]: {user_query_1}")

    # 扣动图网络执行 (Stream 模式实时查看节点轨迹)
    for event in app.stream({"messages": [HumanMessage(content=user_query_1)]}, config=session_config):
        for node_name, node_state in event.items():
            print(f"[Node Finished]: {node_name}")

    # 获取当前会话最终输出
    final_state = app.get_state(session_config)
    last_msg = final_state.values["messages"][-1]
    print(f"[Agent Response]:\n{last_msg.content}")

    print("\n" + "=" * 60)


