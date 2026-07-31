# 自定义 管道节点
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import  BaseModel, Field
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import  RunnableLambda
load_dotenv()


def sensitive_word_filter(text: str) -> str:
    """自定义管道节点 1：敏感词拦截器"""
    print(f"[自定义节点触发] 正在检查大模型输出...")
    if "机密" in text or "内网" in text:
        # 如果发现违规，直接篡改大模型的输出！
        return "[安全网关拦截]：大模型尝试生成违规的内部机密信息，已被系统屏蔽！"
    return text

def add_company_watermark(text: str) -> str:
    """自定义管道节点 2：企业版权水印"""
    return text + "\n\n(© 2024 企业专属 AI 大脑 内部绝密)"


llm = init_chat_model(
    model='xiaomi/mimo-v2.5-pro',
    model_provider='openrouter',
    temperature=0,
    max_tokens= 5000
)
message = ChatPromptTemplate.from_template("请用一句话告诉我关于 {topic} 的事。")
parser = StrOutputParser()


chain = message | llm | parser | RunnableLambda(sensitive_word_filter) | RunnableLambda(add_company_watermark)

print("启动带自定义网关的 LCEL 管道...\n")

# 模拟用户恶意提问套取内网信息
result = chain.invoke({"topic": "公司的内网服务器密码"})
print(result)


