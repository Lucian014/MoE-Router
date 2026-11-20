import { useLayoutEffect, useRef, useState } from "react";
import { ArrowUpIcon, RouteIcon } from "./icons";

/** The message box: Enter sends, Shift+Enter adds a line; disabled while an answer is pending. */
export function Composer({ onSend, busy, autoFocus }: { onSend: (text: string) => void; busy: boolean; autoFocus?: boolean }) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 240)}px`;
  }, [text]);

  const canSend = !busy && text.trim().length > 0;
  const send = () => {
    if (!canSend) return;
    onSend(text.trim());
    setText("");
  };

  return (
    <div className="composer">
      <textarea
        ref={ref}
        rows={1}
        value={text}
        autoFocus={autoFocus}
        placeholder={busy ? "Se generează răspunsul…" : "Scrie un mesaj…"}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            send();
          }
        }}
        aria-label="Mesaj"
      />
      <div className="composer-bottom">
        <span className="routing-mode" title="Routerul alege modelul după domeniul și complexitatea mesajului">
          <RouteIcon size={16} className="accent" /> Rutare automată
        </span>
        <button className="send" onClick={send} disabled={!canSend} aria-label="Trimite">
          <ArrowUpIcon size={18} />
        </button>
      </div>
    </div>
  );
}
