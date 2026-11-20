import { useEffect, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { AssistantMessage as AssistantMsg } from "../conversations";
import { complexityLabel, fmtMs, fragments } from "../format";
import { CheckIcon, CopyIcon, InfoIcon, RetryIcon } from "./icons";

export function UserMessage({ text }: { text: string }) {
  return (
    <div className="turn turn-user">
      <div className="bubble">{text}</div>
    </div>
  );
}

/** The routing decision under an answer: what the user sees without opening the details. */
export function RoutingChips({ message }: { message: AssistantMsg }) {
  const { reply } = message;
  const r = reply.routing;
  const nFragments = r.retrieved_chunk_ids?.length ?? 0;
  return (
    <div className="chips">
      <span className="chip chip-accent" title="Domeniul pe care a fost rutată cererea">
        {r.effective_domain ?? "—"}
      </span>
      <span className="chip" title="Complexitatea pe care a fost rutată cererea">
        complexitate {complexityLabel(r.complexity)}
      </span>
      <span className="chip" title={`Motor: ${reply.engine}`}>
        {reply.model}
      </span>
      {r.inherited && (
        <span className="chip" title="Domeniul și complexitatea vin din tura anterioară">
          moștenit
        </span>
      )}
      {nFragments > 0 && <span className="chip">RAG · {fragments(nFragments)}</span>}
      {reply.truncated && (
        <span className="chip chip-warning" title="Răspunsul a atins limita de tokeni a modelului">
          trunchiat
        </span>
      )}
      <span className="chip" title="Timpul de generare al modelului">
        {fmtMs(reply.latency_ms)}
      </span>
    </div>
  );
}

export function AssistantMessage({
  message,
  selected,
  onDetails,
}: {
  message: AssistantMsg;
  selected: boolean;
  onDetails: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(message.reply.response);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // the clipboard API needs a secure context; nothing to do over plain http
    }
  };
  return (
    <div className="turn turn-assistant">
      <div className="markdown">
        <Markdown remarkPlugins={[remarkGfm]}>{message.reply.response}</Markdown>
      </div>
      <RoutingChips message={message} />
      <div className="actions">
        <button
          className={`icon-button${selected ? " is-active" : ""}`}
          onClick={onDetails}
          title="Detalii rutare"
          aria-label="Detalii rutare"
        >
          <InfoIcon size={16} />
        </button>
        <button className="icon-button" onClick={copy} title="Copiază" aria-label="Copiază răspunsul">
          {copied ? <CheckIcon size={16} /> : <CopyIcon size={16} />}
        </button>
      </div>
    </div>
  );
}

export function ErrorMessage({ text, onRetry, disabled }: { text: string; onRetry: () => void; disabled: boolean }) {
  return (
    <div className="turn turn-assistant">
      <div className="error-box">
        <span>{text}</span>
        <button className="button button-secondary" onClick={onRetry} disabled={disabled}>
          <RetryIcon size={15} /> Reîncearcă
        </button>
      </div>
    </div>
  );
}

/** While /api/generate runs: the routing is only known with the answer (no streaming yet). */
export function PendingAnswer({ startedAt }: { startedAt: number }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const seconds = Math.max(0, Math.floor((now - startedAt) / 1000));
  return (
    <div className="turn turn-assistant pending" aria-live="polite">
      <div className="pending-status">
        <span className="dots" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
        <span>Se generează răspunsul…</span>
        <span className="mono muted">{seconds} s</span>
      </div>
      <div className="skeleton" style={{ width: "100%" }} />
      <div className="skeleton" style={{ width: "92%" }} />
      <div className="skeleton" style={{ width: "60%" }} />
    </div>
  );
}
