# 图书智能体（Library Agent）

面向图书和文档知识库的智能体项目骨架。后续将支持 PDF 导入、父子块切分、混合检索、图式智能体编排，以及基于 FastAPI 的流式对话接口。

> 当前仓库只包含项目结构、依赖与开发配置，尚未实现业务代码，因此暂时不能启动服务。

## 项目结构

```text
.
├── core/                   # 核心配置与业务异常
│   ├── config.py
│   └── exceptions.py
├── infrastructure/         # 第三方基础设施适配
│   ├── llm_client.py
│   ├── vector_store.py
│   └── database.py
├── rag_engine/             # RAG 文档处理与检索
│   ├── document_loader.py
│   ├── text_splitter.py
│   └── retriever.py
├── agent/                  # 图式智能体编排
│   ├── tools/
│   ├── state.py
│   ├── nodes.py
│   └── graph_builder.py
├── api/                    # FastAPI 接口层
│   ├── routes/
│   │   ├── upload.py
│   │   └── chat.py
│   ├── schemas.py
│   └── main.py
├── tests/                  # 后续测试目录
├── .env.example            # 环境变量示例
├── requirements.txt        # Python 依赖
└── Dockerfile              # 容器化占位文件
```

## 目录职责

- `core`：统一配置读取和领域异常。
- `infrastructure`：LLM、向量数据库、SQLite 等外部服务适配。
- `rag_engine`：文档解析、文本切分、向量与关键词混合检索。
- `agent`：工具定义、状态管理、节点和 LangGraph 图组装。
- `api`：请求模型、文件上传和 SSE 对话路由。

## 开发准备

需要 Python 3.11 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 中填写模型服务地址、模型名称和 API 密钥；该文件已被 Git 忽略，不应提交到仓库。
