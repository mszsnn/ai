
# 常规的向量检索，可能会发生检索出来很多重复性的内容， 如果你的文档由多处地方定义了相关的内容
# MMR  Maximum Marginal Relevance 最大边际相关性检索。目标： 找最像的，且候选结果不要太像
import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# 已经有了 词向量数据库了， 直接加载， 但是依旧要模型，因为还需要转化用户提问

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

path = './chatbi_chroma_db'
vectorstore = Chroma(
    persist_directory=path,
    embedding_function=embeddings_model  # 注意这里的参数名变成了 embedding_function
)

print('模型加载完毕 \n')

# 传统检索
query = "帮我查一下那些花钱很多的高级账号的定义，以及跟退货有关的规定。"
search_result = vectorstore.similarity_search(query, k=3)

print('传统检索 ===============\n')
for i, doc in enumerate(search_result):
    print(f"结果 {i + 1}: {doc.page_content[:30]}...")


print('MMR 检索 ===============\n')

mmr_search_result = vectorstore.max_marginal_relevance_search(
    query,
    k=3, # 挑选最好的 3个
    fetch_k=10,  # 取出来 10个
    lambda_mult=0.5 # 表示在最相关和最多样性之间各占比 50%
)

for i, doc in enumerate(mmr_search_result):
    print(f"结果 {i+1}: {doc.page_content[:30]}...")


print('GREP 精准检索 ===============\n')

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# 包装为提取器
vector_retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs = {'k': 3, 'fetch_k': 10, 'lambda_mult': 0.5}
)

# split_result 需要从数据插入和防断裂切分
bm25_retriever  = BM25Retriever.from_documents(split_result)
bm25_retriever.k =3

# 混合的最终拾取器
hybrid_retriever = EnsembleRetriever(
    retrievers = [ vector_retriever, bm25_retriever ],
    weights=[0.5, 0.5]
)
# 混合之后得到结果
docs = hybrid_retriever.invoke(query)








#==============链接大模型

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(
    model='qwen/qwen3.7-flash',
    model_provider='openrouter',
    temperature=0
)

# 组装上下文
context = '\n'.join([item.page_content for item in mmr_search_result])


# 设计带有 RAG 的专属 prompt
prompt_template = ChatPromptTemplate.from_messages([
    ('system', """你是一个专业的公司内部数据分析助手 (ChatBI)。请严格根据以下提供给你的【业务字典参考资料】来回答用户的问题。
如果参考资料中没有相关信息，请直接回答“抱歉，业务字典中未找到相关定义”，绝对不要编造！【业务字典参考资料】:{context}"""),
    ("human", "{question}")
])

workflow = prompt_template | llm | StrOutputParser()


print("大脑正在阅读检索到的资料并思考...\n")

final_answer = workflow.invoke({
    'context': context,
    'question': query
})


print("==================================================")
print(final_answer)
print("==================================================")



