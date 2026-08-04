# 图书智能体（Library Agent）

面向图书和文档知识库的智能体项目骨架。后续将支持 PDF 导入、父子块切分、混合检索、图式智能体编排，以及基于 FastAPI 的流式对话接口。


✅ 第一阶段 (RAG 底座) 全部打通：解析、切分、微批次入库、带溯源的向量检索。

✅ 第二阶段 (智能体大脑) 完美收官：Pydantic 强约束工具、LangGraph 状态机、SQLite 持久化记忆、自动记忆垃圾回收（GC）、API 容错自愈。

✅ 第三阶段：FastAPI 后端封装与 SSE 流式输出
[x] TODO 3.1：搭建 FastAPI 基础架构与依赖注入

创建 api/server.py，配置 FastAPI 应用和 CORS 跨域。

将我们之前写的全局单例 VectorStore 和 BookAgentBuilder 挂载到 FastAPI 的生命周期（Lifespan）中。

[x] TODO 3.2：手撕 SSE (Server-Sent Events) 流式推送接口

编写 /api/chat 接口。

利用 Python 的异步生成器 (AsyncGenerator)，拦截 LangGraph app.stream() 吐出的每一个 Token，通过 SSE 协议实时推给前端。

[x] TODO 3.3：多租户动态建库接口 (文件上传)

编写 /api/upload 接口。

接收用户传来的新 PDF，后台调用你的 VectorPipeline，自动新建 Collection 并打入向量，实现“上传即刻能聊”。

### 第三阶段启动

在仓库根目录安装依赖并启动 API：

```bash
pip install -r library_agent/requirements.txt
cp library_agent/.env.example library_agent/.env
uvicorn library_agent.api.server:app --reload
```

接口约定：

- `GET /healthz`：检查服务是否完成初始化。
- `GET /api/books`：读取持久化书架。
- `POST /api/upload`：使用 `multipart/form-data`，字段为 `book_id` 和 `file`；支持 PDF、TXT，上传完成后才返回，确保可以立即对话。
- `POST /api/chat`：JSON 字段为 `message`、`book_id`，可选 `thread_id`；响应为 SSE，事件包括 `start`、`status`、`token`、`done` 和 `error`。
- `GET /api/books/{book_id}/threads/{thread_id}/messages`：读取当前书籍和会话的历史消息。

书籍元数据和前端展示用的聊天消息保存在 `data/library.sqlite3`，上传文件保存在 `uploads/`，LangGraph 的上下文记忆仍保存在 `agent/checkpoints.sqlite`。这些本地运行数据已加入 `.gitignore`，不会进入 Git 提交。生产部署时需要将 `data/`、`uploads/` 和相关 SQLite 文件挂载到持久化卷。

应用日志统一以 JSON Lines 输出到标准输出，包含 `event`、`request_id`、`book_id`、`thread_id`、耗时和异常堆栈等字段。可通过 `LOG_LEVEL=DEBUG` 临时提高日志级别；Docker 部署时可用以下命令查看后端日志：

```bash
docker compose -f library_agent/docker-compose.yml logs -f backend
```

聊天链路常见事件包括 `chat_started`、`chat_retrieval_started`、`chat_completed`、`chat_failed` 和 `chat_client_disconnected`。

`book_id` 会直接作为 Chroma Collection 名称，必须是 3-63 位、以字母或数字开头和结尾，只能包含字母、数字、下划线和连字符。



✅ 第四阶段： 前端界面与智能体交互 (Generative UI 雏形)

[x] TODO 4.1：构建多租户聊天 UI

左侧边栏：显示“我的书架”（不同的 book_id）。

主聊天区：对接 SSE 接口，实现打字机渲染和 Markdown 解析。

[x] TODO 4.2：精准溯源 (Citation) 渲染

前端解析大模型回答末尾的 【来源 Page X】，将其渲染成可点击的漂亮 Tag，甚至在右侧弹出对应的 PDF 原文页面（增强防幻觉体验）。

前端工程位于 `frontend/`，当前已实现书架、上传、SSE 对话、来源标签和移动端布局。详细启动方式见 `frontend/README.md`。

第五阶段：容器化与云端部署 (Production Deployment)

[x] TODO 5.1：编写工业级 Dockerfile

已完成 Python 后端镜像、非 root 运行用户、健康检查和 `/data` 持久化卷。`/data/chroma`、`/data/uploads`、`/data/library.sqlite3`、`/data/checkpoints.sqlite` 会随容器重启保留。

[x] TODO 5.2：Docker Compose 与 Nginx 反向代理

已完成前端 Node SSR 服务、FastAPI 后端、Nginx 网关和 SSE 代理配置。Nginx 会将 `/api` 转发到后端并关闭响应缓冲，避免流式回答被攒到最后才显示。

### 第五阶段启动

先准备 `library_agent/.env`，至少填写模型和向量服务所需的密钥，然后在仓库根目录执行：

```bash
docker compose -f library_agent/docker-compose.yml up --build
```

浏览器访问 `http://localhost:8080`。停止服务但保留数据：

```bash
docker compose -f library_agent/docker-compose.yml down
```

不要随意使用 `down -v`，它会删除 `library_agent_data` 卷中的书籍、向量库和聊天记录。首次构建会把仓库里的 Chroma 数据作为种子复制到持久化卷；之后容器重启不会覆盖用户新上传的数据。
