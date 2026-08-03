from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.runnables.config import RunnableConfig
from library_agent.infrastructure.vector_store import get_global_vector_store
from library_agent.rag_engine.vector_searching import VectorSearching

class BookSearchKeyword(BaseModel):
    keyword: str = Field(..., description='搜索关键词')

@tool('search_keyword_tool', args_schema=BookSearchKeyword)
def search_keyword_tool(keyword: str, config: RunnableConfig) -> str:
    """
    当用户询问关于本书业务知识、核心概念、或者需要权威资料时候，需要调用此工具
    """
    # config 是穿透传递进来的
    book_id = config.get('configurable', {}).get('book_id')

    if not book_id:
        return '系统错误,未获取当前书本的 ID'

    print(f'调用工具查询书本{book_id}, 关键词：{keyword}\n')

    # 获取全局db 链接
    global_store = get_global_vector_store()

    # 查询实例
    retriever = VectorSearching(vector_store=global_store, tenant_id=book_id)


    # 查询
    results = list(retriever.search(keyword))

    # kon
    if not results:
        return '未能查询到结果，直接告知用户没找到，禁止编纂答案'

    # 最终结果， 返回给 agent llm
    context = '下面是从知识库中检索到的权威原文片段：\n\n'

    for i, (doc, meta, dist) in enumerate(results):
        context = context + f" [Top {i + 1} 匹配] (数学距离: {dist:.4f}): \n"
        context = context + f"绝对坐标溯源: {meta} \n"
        context = context + f"原文片段:\n{doc}\n\n"
    return context

# 后续工具直接扔这里
agent_tools = [
    search_keyword_tool
]