# Library Agent 更新与运维命令

本文档默认项目目录为仓库根目录，Docker Compose 文件位于：

```text
library_agent/docker-compose.yml
```

## 一、提交本地修改

先查看修改内容：

```bash
git status --short
git diff --stat
```

不要直接使用 `git add .`。以下文件是运行时产生的聊天状态，不应提交：

```text
library_agent/agent/checkpoints.sqlite
library_agent/agent/checkpoints.sqlite-shm
library_agent/agent/checkpoints.sqlite-wal
library_agent/data/library.sqlite3
```

根据实际修改添加代码文件，例如：

```bash
git add \
  library_agent/frontend/app \
  library_agent/frontend/sites-vite-plugin.ts \
  library_agent/frontend/vite.config.ts \
  library_agent/api \
  library_agent/agent/graph.py \
  library_agent/agent/tools.py \
  library_agent/deploy/nginx.conf
```

提交并推送：

```bash
git commit -m "update: improve library agent"
git push origin master
```

## 二、服务器拉取最新代码

在服务器进入仓库目录：

```bash
cd /path/to/ai
git pull origin master
```

如果服务器当前已经位于 `library_agent` 目录，则先回到仓库根目录：

```bash
cd ..
git pull origin master
```

## 三、只更新前端

适用于修改 React 页面、CSS、前端构建配置等情况：

```bash
docker compose -f library_agent/docker-compose.yml build --no-cache frontend
docker compose -f library_agent/docker-compose.yml up -d --force-recreate frontend
```

浏览器执行强制刷新：

```text
macOS: Cmd + Shift + R
Windows/Linux: Ctrl + Shift + R
```

## 四、只更新后端

适用于修改 Agent、RAG、API、日志、数据库逻辑等情况：

```bash
docker compose -f library_agent/docker-compose.yml build --no-cache backend
docker compose -f library_agent/docker-compose.yml up -d --force-recreate backend
```

## 五、更新 Nginx 配置

Nginx 配置通过 volume 挂载，不需要重新构建镜像。拉取代码后重新创建 Nginx 容器：

```bash
docker compose -f library_agent/docker-compose.yml up -d --force-recreate nginx
```

检查上传大小限制：

```bash
docker compose -f library_agent/docker-compose.yml exec nginx \
  nginx -T 2>/dev/null | grep client_max_body_size
```

当前配置应为：

```text
client_max_body_size 500m;
```

## 六、前后端和 Nginx 全部更新

```bash
docker compose -f library_agent/docker-compose.yml build --no-cache frontend backend
docker compose -f library_agent/docker-compose.yml up -d --force-recreate frontend backend nginx
```

## 七、查看服务状态

```bash
docker compose -f library_agent/docker-compose.yml ps
```

检查后端健康状态：

```bash
curl http://127.0.0.1:8080/healthz
```

预期返回类似：

```json
{"status":"ok","agent_ready":true}
```

## 八、查看日志

查看后端实时日志：

```bash
docker compose -f library_agent/docker-compose.yml logs -f backend
```

查看最近 200 行后端日志：

```bash
docker compose -f library_agent/docker-compose.yml logs --tail=200 backend
```

查看 Nginx 日志：

```bash
docker compose -f library_agent/docker-compose.yml logs -f nginx
```

重点关注这些后端事件：

```text
upload_started
upload_completed
upload_failed
chat_started
chat_completed
chat_failed
knowledge_search_started
knowledge_search_completed
```

## 九、清理聊天记录和 Agent 上下文

执行前先停止后端，避免 SQLite 被占用：

```bash
docker compose -f library_agent/docker-compose.yml stop backend
```

Docker 部署使用持久化 volume，执行以下命令清理聊天记录和 LangGraph 上下文：

```bash
docker compose -f library_agent/docker-compose.yml run --rm --no-deps backend \
  python -c "import sqlite3; c=sqlite3.connect('/data/library.sqlite3'); c.executescript('DELETE FROM chat_messages; DELETE FROM threads; VACUUM;'); c.close(); c=sqlite3.connect('/data/checkpoints.sqlite'); c.executescript('DELETE FROM writes; DELETE FROM checkpoints; VACUUM;'); c.close()"
```

重新启动后端：

```bash
docker compose -f library_agent/docker-compose.yml up -d backend
```

此操作不会删除：

- 书籍元数据
- 上传的 PDF/TXT 文件
- Chroma 向量库

## 十、本地开发环境清理聊天记录

停止本地 `uvicorn` 后，在仓库根目录执行：

```bash
sqlite3 library_agent/data/library.sqlite3 \
  'DELETE FROM chat_messages; DELETE FROM threads; VACUUM;'

sqlite3 library_agent/agent/checkpoints.sqlite \
  'DELETE FROM writes; DELETE FROM checkpoints; VACUUM;'

rm -f library_agent/agent/checkpoints.sqlite-shm \
      library_agent/agent/checkpoints.sqlite-wal
```

## 十一、停止和启动服务

停止服务但保留数据：

```bash
docker compose -f library_agent/docker-compose.yml down
```

启动已有镜像：

```bash
docker compose -f library_agent/docker-compose.yml up -d
```

首次部署或需要重新构建全部镜像：

```bash
docker compose -f library_agent/docker-compose.yml up -d --build
```

不要随意执行以下命令：

```bash
docker compose -f library_agent/docker-compose.yml down -v
```

`down -v` 会删除持久化 volume，可能导致上传书籍、向量库、聊天记录和 Agent 状态全部丢失。

