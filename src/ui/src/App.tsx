import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, generate } from "./api";
import { Composer } from "./components/Composer";
import { DetailsPanel } from "./components/DetailsPanel";
import { EmptyState } from "./components/EmptyState";
import { InfoIcon, PanelIcon } from "./components/icons";
import { AssistantMessage, ErrorMessage, PendingAnswer, UserMessage } from "./components/Messages";
import { Sidebar } from "./components/Sidebar";
import {
  loadConversations,
  newId,
  saveConversations,
  titleFrom,
  type AssistantMessage as AssistantMsg,
  type Conversation,
} from "./conversations";

type Theme = "light" | "dark";
const THEME_KEY = "llm-router.theme";

function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    // storage blocked: follow the system
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function errorText(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.status === 504) return `Modelul nu a răspuns la timp (504). ${e.message}`;
    if (e.status === 502) return `Routerul sau workerul a răspuns cu o eroare (502). ${e.message}`;
    if (e.status === 503) return "Gateway-ul nu e conectat la RabbitMQ (503). Încearcă din nou peste câteva secunde.";
    return `Eroare ${e.status}: ${e.message}`;
  }
  return "Nu pot contacta serverul. Verifică dacă gateway-ul rulează.";
}

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>(loadConversations);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [pending, setPending] = useState<{ convId: string; startedAt: number } | null>(null);
  const [detailsId, setDetailsId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 760);
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const threadEnd = useRef<HTMLDivElement>(null);

  const active = conversations.find((c) => c.id === activeId) ?? null;
  const assistantMessages = (active?.messages ?? []).filter((m): m is AssistantMsg => m.role === "assistant");
  const detailsMessage = assistantMessages.find((m) => m.id === detailsId) ?? null;
  const busy = pending !== null;

  useEffect(() => saveConversations(conversations), [conversations]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      // the theme is only a convenience
    }
  }, [theme]);

  useEffect(() => {
    threadEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [active?.messages.length, pending]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setDetailsId(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const update = useCallback((id: string, fn: (c: Conversation) => Conversation) => {
    setConversations((list) => list.map((c) => (c.id === id ? fn(c) : c)));
  }, []);

  const send = async (text: string, conv: Conversation | null = active) => {
    if (busy) return;
    let target = conv;
    if (!target) {
      target = { id: newId(), sessionId: null, title: titleFrom(text), updatedAt: Date.now(), messages: [] };
      const created = target;
      setConversations((list) => [created, ...list]);
      setActiveId(created.id);
    }
    const { id: convId, sessionId } = target;
    update(convId, (c) => ({
      ...c,
      updatedAt: Date.now(),
      messages: [...c.messages, { id: newId(), role: "user", text }],
    }));
    const startedAt = Date.now();
    setPending({ convId, startedAt });
    try {
      const reply = await generate(text, sessionId);
      update(convId, (c) => ({
        ...c,
        sessionId: reply.session_id,
        updatedAt: Date.now(),
        messages: [...c.messages, { id: newId(), role: "assistant", reply, clientMs: Date.now() - startedAt }],
      }));
    } catch (e) {
      update(convId, (c) => ({
        ...c,
        messages: [...c.messages, { id: newId(), role: "error", text: errorText(e), prompt: text }],
      }));
    } finally {
      setPending(null);
    }
  };

  /** Drops the failed prompt and its error, then sends the prompt again. */
  const retry = (errorId: string, prompt: string) => {
    if (!active || busy) return;
    const conv = active;
    update(conv.id, (c) => {
      const i = c.messages.findIndex((m) => m.id === errorId);
      const from = i > 0 && c.messages[i - 1].role === "user" ? i - 1 : i;
      return { ...c, messages: c.messages.filter((_, j) => j < from || j > i) };
    });
    void send(prompt, conv);
  };

  const toggleDetails = () => {
    const last = assistantMessages.at(-1);
    setDetailsId(detailsMessage ? null : (last?.id ?? null));
  };

  return (
    <div className={`app${sidebarOpen ? "" : " sidebar-closed"}${detailsMessage ? " panel-open" : ""}`}>
      {sidebarOpen && (
        <Sidebar
          conversations={conversations}
          activeId={activeId}
          onSelect={(id) => {
            setActiveId(id);
            setDetailsId(null);
          }}
          onNew={() => {
            setActiveId(null);
            setDetailsId(null);
          }}
          onCollapse={() => setSidebarOpen(false)}
          theme={theme}
          onToggleTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
        />
      )}

      <main className="main">
        <header className="main-header">
          {!sidebarOpen && (
            <button className="icon-button" onClick={() => setSidebarOpen(true)} aria-label="Arată bara laterală">
              <PanelIcon size={20} />
            </button>
          )}
          <h1 className="main-title">{active?.title ?? ""}</h1>
          {assistantMessages.length > 0 && (
            <button className={`button button-secondary${detailsMessage ? " is-active" : ""}`} onClick={toggleDetails}>
              <InfoIcon size={16} /> Detalii rutare
            </button>
          )}
        </header>

        {active ? (
          <>
            <div className="thread">
              <div className="column thread-inner">
                {active.messages.map((m) =>
                  m.role === "user" ? (
                    <UserMessage key={m.id} text={m.text} />
                  ) : m.role === "assistant" ? (
                    <AssistantMessage
                      key={m.id}
                      message={m}
                      selected={m.id === detailsId}
                      onDetails={() => setDetailsId(m.id === detailsId ? null : m.id)}
                    />
                  ) : (
                    <ErrorMessage key={m.id} text={m.text} disabled={busy} onRetry={() => retry(m.id, m.prompt)} />
                  ),
                )}
                {pending?.convId === active.id && <PendingAnswer startedAt={pending.startedAt} />}
                <div ref={threadEnd} />
              </div>
            </div>
            <div className="composer-area">
              <div className="column">
                <Composer onSend={(t) => void send(t)} busy={busy} />
              </div>
              <p className="footnote">Modelul e ales automat de router, după domeniul și complexitatea mesajului.</p>
            </div>
          </>
        ) : (
          <EmptyState onSend={(t) => void send(t)} busy={busy} />
        )}
      </main>

      {detailsMessage && <DetailsPanel message={detailsMessage} onClose={() => setDetailsId(null)} />}
    </div>
  );
}
