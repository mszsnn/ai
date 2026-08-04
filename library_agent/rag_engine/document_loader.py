

# 加载 txt 文本 pdf 目前先支持这两种方式
# 1. 但是这两种方式需要支持统一的输出结构，pdf 天然是有页码的， 但是 txt 没有，所以需要需要构建 txt 的虚拟页码
# 2. 绝对不是直接加载全部的文本或者 pdf， 肯定是分段加载的
from langchain_text_splitters import RecursiveCharacterTextSplitter
from abc import ABC, abstractmethod
import fitz
from typing import Iterator, Dict, Any

# ==========================================
# 定义解析契约，未来其他文件格式的解析，也必须按照这种格式来
# ==========================================

class BaseDocumentParser(ABC):
    """ 文档解析基类， 强制后续的子类，必须实现 stream 方式的读取，并且返回标准结构"""
    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def text_stream(self) -> Iterator[Dict[str, Any]]:
        """
        期望 yield 下面的格式

        {
            "content": "提取出的文本",
            "metadata": {"source_type": "pdf/txt", "loc": "页码"}
        }

        """
        # 抽象类不能为空， 占位
        pass



# ==========================================
# pdf 和 TXT 解析
# ==========================================

def get_pdf_page_count(file_path: str) -> int:
    """Return the page count and fail clearly for password-protected PDFs."""
    pdf_doc = fitz.open(file_path)
    try:
        if pdf_doc.needs_pass:
            raise ValueError("PDF 受密码保护，请先解除密码后再上传")
        return pdf_doc.page_count
    finally:
        pdf_doc.close()

class PDFParser(BaseDocumentParser):
    """ 基于原始页码解析 """
    def text_stream(self) -> Iterator[Dict[str, Any]]:
        pdf_doc = fitz.open(self.file_path)

        if pdf_doc.needs_pass:
            pdf_doc.close()
            raise ValueError("PDF 受密码保护，请先解除密码后再上传")

        for page_number, page in enumerate(pdf_doc, 1):
            # 要纯文本输出
            text = page.get_text('text')
            # 有值返回
            if text.strip():
                yield {
                    'content': text,
                    'metadata': {"source_type": "pdf", "loc": f'Page {page_number}'}
                }
        pdf_doc.close()


class TXTParser(BaseDocumentParser):
    """ 构造虚拟页码 """
    def text_stream(self) -> Iterator[Dict[str, Any]]:
        with open(self.file_path, 'r', encoding='utf-8') as file:
            # 临时行存储
            batch_lines = []
            start_line = 1

            for current_line, line in enumerate(file, 1):
                batch_lines.append(line)
                # 50 行 为一页
                if len(batch_lines) >= 50:
                    yield {
                        'content': ''.join(batch_lines),
                        'metadata': {"source_type": "txt", "loc": f'Lines {start_line}-{current_line}'}
                    }

                    batch_lines = []
                    start_line = current_line + 1

            if batch_lines:
                yield {
                    "content": "".join(batch_lines),
                    "metadata": {"source_type": "txt", "loc": f"Lines {start_line}-{current_line}"}
                }


# ==========================================
# 核心流水线， 进行文本拆分
# ==========================================


class TextProcessor:
    """ 文本拆分器 """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100 ):
        self.split_exec = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            ## 按照语义大小尝试拆分
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
        )


    def stream_chunks(self, page_stream: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """
        将之前的每一页进行拆分成数据 chunk, 后续存储到数据库中
        """

        for data in page_stream:
            chunks = self.split_exec.split_text(data['content'])

            for chunk_text in chunks:
                # 这里有一个非常深的的做法， 为啥不直接， 将一整页的数据直接去存储到向量数据库中
                # vectorstore = Chroma.from_documents(
                #     documents=split_result,
                #     embedding=embedding_model,
                #     persist_directory=persist_directory

                # 工程优化，每页的 chunk 不定，有可能2个，有可能 100个，我们划分成单 chunk 的话可以人为固定
                # 100个chunk 发一次api 模型请求

                yield {
                    "text": chunk_text,
                    "metadata": data["metadata"]
                }




if __name__ == "__main__":
    import os

    def get_parser(file_path: str) -> BaseDocumentParser:
        if file_path.lower().endswith('.pdf'):
            return PDFParser(file_path)
        elif file_path.lower().endswith('.txt'):
            return TXTParser(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")


    # -------- 测试区 --------
    test_file = "./description.txt"  # 切换后缀试试！

    if os.path.exists(test_file):
        print(f"流式解析管道，正在处理: {test_file}")

        # 1. 动态获取解析器并初始化处理器
        parser = get_parser(test_file)
        processor = TextProcessor(chunk_size=500, chunk_overlap=50)

        # 2. 组装数据流水线
        data_stream = parser.text_stream()
        chunk_stream = processor.stream_chunks(data_stream)

        # 3. 扣动扳机拉取数据
        for i, chunk_data in enumerate(chunk_stream):
            print(f"\n--- 📦 Chunk {i + 1} ---")
            print(f"坐标: {chunk_data['metadata']}")
            print(f"文本: {chunk_data['text'][:100]}...\n")

            if i >= 2:
                break
        print("多格式 Metadata 完美对齐")
    else:
        print(f"请在当前目录放入一个测试文件 {test_file} 后重试。")
