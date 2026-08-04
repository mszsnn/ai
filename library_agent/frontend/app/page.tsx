"use client";

import {
  ArrowUp,
  BookOpen,
  Check,
  FileText,
  FolderOpen,
  LibraryBig,
  LoaderCircle,
  Menu,
  MoreHorizontal,
  Paperclip,
  Plus,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  UploadCloud,
  X,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ChangeEvent,
  FormEvent,
  KeyboardEvent,
  useMemo,
  useRef,
  useState,
  useEffect,
} from "react";

type Book = {
  id: string;
  title: string;
  meta: string;
  color: string;
  accent: string;
  indexed?: boolean;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
};

type SseFrame = {
  event: string;
  data: Record<string, string>;
};

const API_BASE =
  typeof process !== "undefined" && process.env.NEXT_PUBLIC_API_BASE_URL
    ? process.env.NEXT_PUBLIC_API_BASE_URL
    : "";

const starterMessages: Message[] = [
  {
    id: "welcome",
    role: "assistant",
    content:
      "你好，我已经准备好和你一起阅读这本书。\n\n你可以问我概念、原则、实践方式，也可以让我根据书中的原文进行对比和总结。每个答案都会尽量带上页码出处。",
  },
];

const bookPalette = [
  { color: "#dbe8ff", accent: "#4d6fff" },
  { color: "#e7defe", accent: "#8765db" },
  { color: "#dcefe5", accent: "#4d9a7a" },
  { color: "#f4e3cf", accent: "#c27e45" },
];

type RemoteBook = Pick<Book, "id" | "title" | "meta" | "indexed">;

function decorateBook(book: RemoteBook, index: number): Book {
  const palette = bookPalette[index % bookPalette.length];
  return { ...book, ...palette };
}

function slugifyFilename(filename: string) {
  const withoutExtension = filename.replace(/\.[^/.]+$/, "");
  const slug = withoutExtension
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 56);
  return (slug || "new_book").padEnd(3, "_book").slice(0, 63);
}

function parseSseFrame(rawFrame: string): SseFrame | null {
  const lines = rawFrame.split("\n");
  let event = "message";
  let rawData = "";

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      rawData += line.slice(5).trim();
    }
  }

  if (!rawData) return null;

  try {
    return { event, data: JSON.parse(rawData) as Record<string, string> };
  } catch {
    return null;
  }
}

function extractCitations(content: string) {
  return Array.from(new Set(content.match(/Page\s+\d+/gi) ?? []));
}

function getPreviewTitle(title: string) {
  const normalized = title.trim() || "Untitled book";
  const words = normalized.split(/\s+/);
  if (words.length > 1) {
    return { firstLine: words[0], secondLine: words.slice(1).join(" ") };
  }

  const midpoint = Math.ceil(normalized.length / 2);
  return {
    firstLine: normalized.slice(0, midpoint),
    secondLine: normalized.slice(midpoint),
  };
}

function getSourcePreview(book?: Book) {
  const meta = book?.meta ?? "SOURCE";
  const [type = "SOURCE", detail = ""] = meta.split("·").map((part) => part.trim());
  const pageMatch = detail.match(/(\d+)\s+pages?/i);
  return {
    type: type.toUpperCase(),
    detail: detail ? detail.toUpperCase() : "BOOK SOURCE",
    page: pageMatch?.[1] ?? "—",
    title: getPreviewTitle(book?.title ?? "Untitled book"),
    indexed: book?.indexed ?? false,
  };
}

function MessageBubble({ message }: { message: Message }) {
  const citations = message.role === "assistant" ? extractCitations(message.content) : [];

  return (
    <article className={`message-row ${message.role}`}>
      {message.role === "assistant" ? (
        <div className="message-avatar assistant-avatar">
          <Sparkles size={15} strokeWidth={2.2} />
        </div>
      ) : null}
      <div className="message-stack">
        <div className="message-label">
          {message.role === "assistant" ? "LIBRARY AGENT" : "YOU"}
        </div>
        <div className={`message-bubble ${message.role}`}>
          {message.content ? (
            <div className="message-copy markdown-copy">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
            </div>
          ) : (
            <div className="typing-dots" aria-label="正在生成回答">
              <i />
              <i />
              <i />
            </div>
          )}
        </div>
        {citations.length > 0 ? (
          <div className="citation-row">
            {citations.map((citation) => (
              <span className="citation-chip" key={`${message.id}-${citation}`}>
                <FileText size={12} />
                {citation}
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </article>
  );
}

export default function Home() {
  const [books, setBooks] = useState<Book[]>([]);
  const [selectedBookId, setSelectedBookId] = useState("");
  const [messages, setMessages] = useState<Message[]>(starterMessages);
  const [draft, setDraft] = useState("");
  const [search, setSearch] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messageListRef = useRef<HTMLDivElement>(null);
  const threadIdsRef = useRef<Record<string, string>>({});

  function getThreadId(bookId: string) {
    return (threadIdsRef.current[bookId] ??= `web-${bookId}`);
  }

  useEffect(() => {
    let cancelled = false;

    async function loadShelf() {
      try {
        const response = await fetch(`${API_BASE}/api/books`);
        if (!response.ok) throw new Error("书架暂时不可用");
        const remoteBooks = (await response.json()) as RemoteBook[];
        if (cancelled || remoteBooks.length === 0) return;

        setBooks(remoteBooks.map(decorateBook));
        setSelectedBookId(remoteBooks[0].id);
      } catch {
        // Keep the shelf empty until the API is available.
      }
    }

    void loadShelf();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedBookId) return;

    let cancelled = false;
    const threadId = getThreadId(selectedBookId);

    async function loadHistory() {
      setMessages(starterMessages);
      try {
        const response = await fetch(
          `${API_BASE}/api/books/${encodeURIComponent(selectedBookId)}/threads/${encodeURIComponent(threadId)}/messages`,
        );
        if (!response.ok) throw new Error("历史记录暂时不可用");
        const history = (await response.json()) as Message[];
        if (!cancelled && history.length > 0) setMessages(history);
      } catch {
        // A new thread simply starts with the welcome message.
      }
    }

    void loadHistory();
    return () => {
      cancelled = true;
    };
  }, [selectedBookId]);

  useEffect(() => {
    const messageList = messageListRef.current;
    if (!messageList) return;
    messageList.scrollTop = messageList.scrollHeight;
  }, [messages]);

  const activeBook = books.find((book) => book.id === selectedBookId) ?? books[0];
  const sourcePreview = getSourcePreview(activeBook);
  const filteredBooks = useMemo(
    () =>
      books.filter((book) =>
        `${book.title} ${book.id}`.toLowerCase().includes(search.toLowerCase()),
      ),
    [books, search],
  );

  function selectBook(bookId: string) {
    setSelectedBookId(bookId);
    setMobileSidebarOpen(false);
  }

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    const bookId = slugifyFilename(file.name);
    const formData = new FormData();
    formData.append("book_id", bookId);
    formData.append("file", file);
    setUploadStatus(`Indexing ${file.name}…`);

    try {
      const response = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: formData,
      });
      const result = (await response.json()) as { detail?: string; chunks?: number; pages?: number };
      if (!response.ok) {
        throw new Error(result.detail || "Upload failed");
      }

      const nextBook: Book = {
        id: bookId,
        title: file.name.replace(/\.[^/.]+$/, ""),
        meta: result.pages
          ? `${file.name.toUpperCase().split(".").pop()} · ${result.pages} pages`
          : `${file.name.toUpperCase().split(".").pop()} · indexed`,
        color: "#e7defe",
        accent: "#8765db",
        indexed: true,
      };
      setBooks((current) => [nextBook, ...current.filter((book) => book.id !== bookId)]);
      selectBook(bookId);
      setUploadStatus(`${nextBook.title} is ready to read`);
      window.setTimeout(() => setUploadStatus(null), 3800);
    } catch (error) {
      setUploadStatus(error instanceof Error ? error.message : "Upload failed");
    }
  }

  function updateAssistantMessage(messageId: string, content: string, streaming = true) {
    setMessages((current) =>
      current.map((message) =>
        message.id === messageId ? { ...message, content, streaming } : message,
      ),
    );
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const question = draft.trim();
    if (!question || isStreaming || !activeBook) return;

    const assistantId = `assistant-${Date.now()}`;
    setMessages((current) => [
      ...current,
      { id: `user-${Date.now()}`, role: "user", content: question },
      { id: assistantId, role: "assistant", content: "", streaming: true },
    ]);
    setDraft("");
    setIsStreaming(true);
    const threadId = getThreadId(activeBook.id);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          book_id: activeBook.id,
          thread_id: threadId,
          message: question,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error("The library agent is unavailable");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let answer = "";

      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";

        for (const rawFrame of frames) {
          const frame = parseSseFrame(rawFrame);
          if (!frame) continue;

          if (frame.event === "token") {
            answer += frame.data.content ?? "";
            updateAssistantMessage(assistantId, answer);
          } else if (frame.event === "status") {
            // Status events are kept for the stream protocol, but are not rendered in the compact UI.
          } else if (frame.event === "error") {
            throw new Error(frame.data.message || "The agent could not finish");
          }
        }

        if (done) break;
      }

      updateAssistantMessage(assistantId, answer, false);
    } catch (error) {
      updateAssistantMessage(
        assistantId,
        error instanceof Error ? error.message : "Something went wrong. Please try again.",
        false,
      );
    } finally {
      setIsStreaming(false);
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  return (
    <main className="library-shell">
      <input
        ref={fileInputRef}
        className="visually-hidden"
        type="file"
        accept=".pdf,.txt,application/pdf,text/plain"
        onChange={handleUpload}
      />

      <aside className={`sidebar ${mobileSidebarOpen ? "is-open" : ""}`}>
        <div className="brand-lockup">
          <div className="brand-mark">
            <LibraryBig size={19} />
          </div>
          <div>
            <div className="brand-name">LIBRARY</div>
            <div className="brand-subtitle">knowledge companion</div>
          </div>
          <button className="icon-button sidebar-close" onClick={() => setMobileSidebarOpen(false)} aria-label="关闭书架">
            <X size={18} />
          </button>
        </div>

        <div className="library-heading">
          <div>
            <span className="section-kicker">YOUR SHELF</span>
            <h2>我的书架</h2>
          </div>
          <button className="add-book-button" onClick={() => fileInputRef.current?.click()} aria-label="上传新书">
            <Plus size={16} />
          </button>
        </div>

        <label className="search-field">
          <Search size={15} />
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search your shelf"
            aria-label="搜索书架"
          />
          <span className="search-shortcut">⌘ K</span>
        </label>

        <div className="book-list">
          {filteredBooks.map((book) => (
            <button
              className={`book-item ${book.id === selectedBookId ? "is-active" : ""}`}
              key={book.id}
              onClick={() => selectBook(book.id)}
            >
              <div className="book-cover" style={{ background: book.color, color: book.accent }}>
                <BookOpen size={18} />
                <span>BK</span>
              </div>
              <span className="book-item-copy">
                <strong>{book.title}</strong>
                <small>{book.meta}</small>
              </span>
              {book.indexed ? <span className="indexed-dot" aria-label="已完成索引" /> : null}
            </button>
          ))}
          {filteredBooks.length === 0 ? <div className="empty-shelf">No books match that search.</div> : null}
        </div>

        <button className="upload-card" onClick={() => fileInputRef.current?.click()}>
          <span className="upload-icon"><UploadCloud size={18} /></span>
          <span>
            <strong>Add a source</strong>
            <small>PDF or TXT · instant indexing</small>
          </span>
          <ArrowUp size={15} className="upload-arrow" />
        </button>

        <div className="sidebar-bottom">
          <div className="privacy-note">
            <ShieldCheck size={15} />
            <span>Your sources stay yours.</span>
          </div>
          <div className="profile-row">
            <div className="profile-avatar">读</div>
            <div>
              <strong>志同道合的读者</strong>
              <small>Local workspace</small>
            </div>
            <MoreHorizontal size={17} className="muted-icon" />
          </div>
        </div>
      </aside>

      {mobileSidebarOpen ? <button className="sidebar-scrim" onClick={() => setMobileSidebarOpen(false)} aria-label="关闭菜单" /> : null}

      <section className="workspace">
        <div className="workspace-grid">
          <section className="chat-column">
            <div className="mobile-chatbar">
              <button className="icon-button mobile-menu" onClick={() => setMobileSidebarOpen(true)} aria-label="打开书架">
                <Menu size={19} />
              </button>
            </div>
            <div className="welcome-panel">
              <div className="welcome-orbit orbit-one" />
              <div className="welcome-orbit orbit-two" />
              <div className="welcome-content">
                <div className="eyebrow"><span className="eyebrow-line" /> CURRENT READING SPACE</div>
                <h1>读得更深，<em>答案自带出处。</em></h1>
                <p>向你的书提问。Library Agent 会先检索原文，再给你一个有根据的回答。</p>
              </div>
              <div className="welcome-numeral">01</div>
            </div>

            <div className="message-list" ref={messageListRef}>
              {messages.map((message) => <MessageBubble key={message.id} message={message} />)}
            </div>

            <form className="composer" onSubmit={handleSubmit}>
              <div className="composer-topline">
                <Sparkles size={15} />
                <span>Ask about {activeBook?.title ?? "your library"}</span>
                <span className="composer-mode">Evidence mode</span>
              </div>
              <textarea
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                placeholder="Ask a question about this book…"
                rows={2}
                disabled={isStreaming}
              />
              <div className="composer-actions">
                <button type="button" className="composer-tool" aria-label="附加文件" onClick={() => fileInputRef.current?.click()}>
                  <Paperclip size={16} />
                  <span>Attach source</span>
                </button>
                <span className="composer-hint">Shift + Enter for a new line</span>
                <button className="send-button" type="submit" disabled={isStreaming || !draft.trim()} aria-label="发送问题">
                  {isStreaming ? <LoaderCircle size={17} className="spin" /> : <Send size={17} />}
                </button>
              </div>
            </form>
          </section>

          <aside className="reference-column">
            <header className="topbar">
              <div className="topbar-right">
                <span className="connection-dot" />
                <span className="connection-label">Agent online</span>
                <button className="icon-button" aria-label="更多操作"><MoreHorizontal size={17} /></button>
              </div>
            </header>

            <div className="reference-content">
              <div className="reference-header">
                <div>
                  <span className="section-kicker">SOURCE TRAIL</span>
                  <h2>阅读上下文</h2>
                </div>
                <button className="icon-button" aria-label="打开来源文件"><FolderOpen size={17} /></button>
              </div>

              <div className="source-preview-card">
                <div className="source-preview-top">
                  <span className="source-type">{sourcePreview.type} / {sourcePreview.indexed ? "INDEXED" : "PROCESSING"}</span>
                  {sourcePreview.indexed ? <Check size={15} /> : <LoaderCircle size={15} className="spin" />}
                </div>
                <div className="source-paper">
                  <div className="paper-topline">{activeBook?.title?.toUpperCase() ?? "BOOK"}</div>
                  <div className="paper-title">
                    {sourcePreview.title.firstLine}
                    {sourcePreview.title.secondLine ? <><br /><span>{sourcePreview.title.secondLine}</span></> : null}
                  </div>
                  <div className="paper-rule" />
                  <div className="paper-caption">{sourcePreview.detail}</div>
                  <div className="paper-page">{sourcePreview.page}</div>
                </div>
                <div className="source-meta-row">
                  <div>
                    <small>ACTIVE SOURCE</small>
                    <strong>{activeBook?.title ?? "Untitled book"}</strong>
                  </div>
                  <span className="source-ready"><span className="status-pulse" /> {sourcePreview.indexed ? "Ready" : "Indexing"}</span>
                </div>
              </div>

              <div className="insight-card">
                <div className="insight-icon"><Sparkles size={16} /></div>
                <div>
                  <span className="section-kicker">READING TIP</span>
                  <p>试试让 Agent 比较两段原文，或追问它的答案有哪些证据支持。</p>
                </div>
              </div>
            </div>
          </aside>
        </div>
      </section>

      {uploadStatus ? (
        <div className="toast" role="status">
          {uploadStatus.startsWith("Indexing") ? <LoaderCircle size={16} className="spin" /> : <Check size={16} />}
          <span>{uploadStatus}</span>
          <button onClick={() => setUploadStatus(null)} aria-label="关闭提示"><X size={14} /></button>
        </div>
      ) : null}
    </main>
  );
}
