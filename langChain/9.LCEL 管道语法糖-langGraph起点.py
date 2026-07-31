
# 在8 里面已经切实体会到：这种langChian 的处理方式完全适配不了大模型的能力边界
# 1 不知道大模型到底需要询问多少遍才能收敛
# 2 一旦开始： 不能暂停和继续

# 所以才会有后续的 langGraph 状态机执行
# 状态机执行之前，先了解核心概念  LCEL LangChain Expression Language 的管道语法糖


from langchain_core.prompts import  ChatPromptTemplate
from langchain.chat_models import  init_chat_model
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 模版， 构造 message
prompt = ChatPromptTemplate.from_template('请用依据简单的话解释什么是 {context}')
# 大模型, 输入 message 输出 AIMessage 对象
llm = init_chat_model(
    model='xiaomi/mimo-v2.5-pro',
    model_provider='openrouter',
    temperature=0,
    max_tokens= 5000
)

# 输出解析， 输入 AIMessage 输出解析字符串
# 大模型输出的是带有 id content token_usage 等等，很多字段， 我们一般只需要 content 纯文本
parse = StrOutputParser()


# 用管道符，将这三步缝合起来， 前者的输出，作为后者的输入

chain = prompt | llm | parse

print("管道执行启动 \n")
result = chain.invoke({"context": "LCEL"})
print("最终输出类型:", type(result))
print("最终输出内容:", result)

