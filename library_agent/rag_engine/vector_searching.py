# 检索

class VectorSearching:
    """ 知识检索 """

    def __init__(self, vector_store, tenant_id:str ):
        self.tenant_id = tenant_id
        # 拿到检索空间
        self.collection = vector_store.get_collection_by_tenant_id(tenant_id=tenant_id)

    def search(self, keyword:str, top_k:int =3):
        """
        核心检索逻辑： 自然语言转化为向量， 进行 cos 相似匹配
        """
        print(f"正在进行匹配： 在知识库 {self.tenant_id}中查找: '{keyword}' ")

        results = self.collection.query(
            query_texts = [keyword],
            n_results=top_k
        )

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        return zip(documents, metadatas, distances)


if __name__ == '__main__':
    # 1. 启动全局向量管家
    from library_agent.infrastructure.vector_store import MultiTenantVectorStore
    vs = MultiTenantVectorStore()

    # 2. 实例化检索
    book_id = "agile_project_management"
    retriever = VectorSearching(vector_store=vs, tenant_id=book_id)

    # 3. 问题
    question = "什么是敏捷开发的核心价值观？"

    # 4. 执行 这里先不考虑混合检索
    search_results = retriever.search(keyword=question, top_k=3)

    # 5. 优雅地打印出捞出来的“海鲜”
    print(f"用户: {question}")

    for i, (doc, meta, dist) in enumerate(search_results):
        print(f" [Top {i + 1} 匹配] (数学距离: {dist:.4f})")
        print(f"绝对坐标溯源: {meta}")
        print(f"原文片段:\n{doc}\n")

    print("检索闭环完成，大模型 RAG 上下文组装完毕！")