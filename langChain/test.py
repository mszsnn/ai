from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from rich import print as p

load_dotenv()


llm = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider='openrouter',
    temperature= 0,
)

msg = [
    SystemMessage(content='你是一个数学家'),
    HumanMessage(content='1 + 1 = ？'),
]

resp = llm.invoke(msg)

# json_data = resp.model_dump()
# json_str = json.dumps(
#     json_data,
#     ensure_ascii=False,
#     indent=2
# )


p(resp)