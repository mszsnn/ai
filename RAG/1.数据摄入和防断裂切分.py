# 数据摄入和切分器

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich import print as p

print('第一步： 加载原始文档')
path = 'description.txt'
loader = TextLoader(path, encoding='utf-8')

docs = loader.load()


# 文本拆分， 当文件很大的时候， 不开课鞥一次性将整个文档喂给ai，所以需要切碎，要设置重叠区，否则的话，可能切断语义
p('进行切分\n')

text_split_executor = RecursiveCharacterTextSplitter(
    chunk_size=100,    # 每个文本块 最大 100个字符
    chunk_overlap=20, # 每个文本块， 20个字符的重叠区
    length_function=len, # 基础的len 计算字符长度
    separators=["\n\n", "\n", "。", "，", " "]  # 切分优先级，先按照段落气氛，不行再按照句子，
)
# 执行
split_result = text_split_executor.split_documents(docs)

####################### 进行向量化构建 ######################################

from langchain_openai import  OpenAIEmbeddings
from langchain_community.vectorstores import  Chroma
from dotenv import  load_dotenv

load_dotenv()

# 实例化模型 专门用来将文本转化为 语义向量， text-embedding-3-small embedding 模型， 已经训练好了参数，
# 用来将语义相近的东西，映射到向量空间附近

embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')


# 将拆分好的文本 ， 向量化之后，存储起来
persist_directory = "./chatbi_chroma_db"

vectorstore = Chroma.from_documents(
    documents=split_result,
    embedding=embedding_model,
    persist_directory=persist_directory
)

# 验证是否存储成功
print(f"成功将 {vectorstore._collection.count()} 个 Chunk 存入向量数据库！\n")


# 假设用户用大白话提问，根本没用你们公司的黑话
query = "帮我查一下那些花钱很多的高级账号的定义"

# 调用数据库底层算法，找出空间距离最近的 2 个卡片 (Top-K)
retrieved_docs = vectorstore.similarity_search(query, k=2)

print(f"‍用户提问: {query} \n")
print("向量数据库召回的最相关文档: \n")
print("--------------------------------------------------")
print(retrieved_docs[0].page_content)
print("--------------------------------------------------")