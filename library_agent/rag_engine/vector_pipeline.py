# 真正将文本存储到数据库中

from library_agent.infrastructure.vector_store import MultiTenantVectorStore
from library_agent.rag_engine.document_loader import (
    PDFParser,
    TXTParser,
    TextProcessor,
)

class VectorPipeline:

    def __init__(self, vector_store: MultiTenantVectorStore):
        # 实例化文档处理器
        self.textProcesser = TextProcessor(chunk_size=500, chunk_overlap=100)
        # 存储 db 示例
        self.vector_store = vector_store

    # 文件区分，获取文件处理 编译器
    def _get_parser(self, file_path: str):
        """动态路由工厂：根据后缀分发不同的解析策略"""
        if file_path.lower().endswith('.pdf'):
            return PDFParser(file_path)
        elif file_path.lower().endswith('.txt'):
            return TXTParser(file_path)
        else:
            raise ValueError(f"系统暂不支持的文件格式: {file_path}")

    # 核心函数

    def book_to_store_by_stream(self, file_path:str, book_id:str, batch_size: int = 100):
        # 先拿到 集合空间，么有就创建
        collection = self.vector_store.create_collection_by_tenant_id(book_id)

        # 得到文件加载器和文件加载流
        # 1 不同文件处理器
        file_parser = self._get_parser(file_path)
        # 2. 单页面流
        page_stream = file_parser.text_stream()
        # 3 切片 chunk 流
        chunk_stream = self.textProcesser.stream_chunks(page_stream)

        # 按照原定计划， 大约 100个chunk 我们发送一次api 请求，将 chunk 转化为 向量，
        batch_texts, batch_metadatas, batch_ids= [], [], []

        global_chunk_index = 0  # 总 chunk 计数
        batch_index = 0  # 输出日志计数

        for chunk in chunk_stream:
            # {
            #     "text": chunk_text,
            #     "metadata": data["metadata"]
            # }

            batch_texts.append(chunk['text'])
            batch_metadatas.append(chunk['metadata'])
            batch_ids.append(f'{book_id}_chunk_{global_chunk_index}')
            global_chunk_index = global_chunk_index + 1

            # 满 100 个发车
            if len(batch_texts) >=batch_size:
                collection.upsert(
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                batch_index = batch_index + 1
                print(f"已成功写入第 {batch_index} 批，累计 {global_chunk_index} 个知识块...\n")
                # 清空
                batch_texts, batch_metadatas, batch_ids= [], [], []

        # 剩余的
        if batch_texts:
            collection.upsert(
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

        print(f"已成功写入最后一批次，累计 {global_chunk_index} 个知识块...\n")



if __name__ == '__main__':
    test_file = "敏捷项目管理.pdf"
    test_book_id = "agile_project_management"

    # 1. 启动全局向量
    vs = MultiTenantVectorStore()

    # 2. 启动流水线
    pipeline = VectorPipeline(vector_store=vs)

    # 3. 开始注水建库
    pipeline.book_to_store_by_stream(file_path=test_file, book_id=test_book_id, batch_size=5)

    # 4. 验证
    collection = vs.get_collection_by_tenant_id(test_book_id)
    count = collection.count()
    print(f"验证成功！目前共有 {count} 条属于 {test_book_id} 的知识块。")

