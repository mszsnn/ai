# Library Agent 前端

React + TypeScript 前端，连接同级目录的 FastAPI Library Agent 后端。

当前页面包含：

- 书架侧栏和本地书籍筛选；
- PDF/TXT 上传并触发向量建库；
- SSE 流式对话和打字机效果；
- `Page X` 来源引用标签；
- 移动端抽屉式书架布局。
- 从后端恢复书架和每本书的固定会话历史，刷新浏览器不会清空。

## 启动

需要 Node.js `>=22.13.0`。先启动后端：

```bash
cd ..
uvicorn library_agent.api.server:app --reload
```

另开终端启动前端：

```bash
npm install
npm run dev
```

开发环境默认通过 Vite 的同源代理访问 `http://127.0.0.1:8000`，因此浏览器不会因为 localhost/127.0.0.1 差异触发 CORS。若后端地址不同，可以设置：

```bash
LIBRARY_API_URL=http://127.0.0.1:8000
```

如果前端部署后与后端不在同一个域名，再设置客户端变量：

```bash
NEXT_PUBLIC_API_BASE_URL=https://your-api.example.com
```

前端为每本书使用稳定的会话 ID（`web-{book_id}`），历史记录由后端 SQLite 保存，不使用浏览器 `localStorage` 作为数据源。

## 验证

```bash
npm run build
npm run lint
npm test
```

页面的组件和视觉语言按 shadcn/ui 的简洁、可组合思路组织；当前仓库没有可安装的 Codex `shadcn/ui` skill，因此没有伪造安装状态。
