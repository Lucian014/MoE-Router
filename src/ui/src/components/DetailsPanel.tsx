import type { ReactNode } from "react";
import type { AssistantMessage } from "../conversations";
import {
  classifierInputLabel,
  complexityLabel,
  fmtInt,
  fmtMs,
  fmtProb,
  routingReasonLabel,
  sessionRoutingLabel,
} from "../format";
import { XIcon } from "./icons";

const TOP_DOMAINS = 5;

function Section({ title, note, children }: { title: string; note?: string; children: ReactNode }) {
  return (
    <section className="panel-section">
      <h3>
        {title}
        {note && <span className="chip chip-accent">{note}</span>}
      </h3>
      {children}
    </section>
  );
}

function Row({ label, value, title }: { label: string; value: ReactNode; title?: string }) {
  return (
    <div className="kv" title={title}>
      <span>{label}</span>
      <span className="mono">{value}</span>
    </div>
  );
}

function Bar({ label, value, highlight }: { label: string; value: number; highlight?: boolean }) {
  return (
    <div className="bar-row">
      <span className="mono bar-label">{label}</span>
      <span className="bar-track">
        <span className={`bar-fill${highlight ? " is-accent" : ""}`} style={{ width: `${Math.max(1, value * 100)}%` }} />
      </span>
      <span className="mono bar-value">{fmtProb(value)}</span>
    </div>
  );
}

/** Everything the router reported for one answer (the `routing` object of /api/generate). */
export function DetailsPanel({ message, onClose }: { message: AssistantMessage; onClose: () => void }) {
  const { reply, clientMs } = message;
  const r = reply.routing;

  const probs = Object.entries(r.task_type_probabilities ?? {}).sort((a, b) => b[1] - a[1]);
  const top = probs.slice(0, TOP_DOMAINS);
  const rest = probs.slice(TOP_DOMAINS).reduce((sum, [, p]) => sum + p, 0);
  const ids = r.retrieved_chunk_ids ?? [];
  const ret = r.retrieval;

  return (
    <aside className="panel" aria-label="Detalii rutare">
      <header className="panel-header">
        <h2>Detalii rutare{reply.turn != null && ` · tura ${reply.turn}`}</h2>
        <button className="icon-button" onClick={onClose} aria-label="Închide">
          <XIcon size={18} />
        </button>
      </header>

      <div className="panel-body">
        <Section title="Decizie">
          <Row label="Domeniu" value={r.effective_domain ?? "—"} />
          <Row label="Complexitate" value={complexityLabel(r.complexity)} />
          <Row label="Motiv (sesiune)" value={sessionRoutingLabel[r.session_routing ?? ""] ?? r.session_routing ?? "—"} title={r.session_routing} />
          <Row label="Alegerea modelului" value={routingReasonLabel[r.routing_reason ?? ""] ?? r.routing_reason ?? "—"} title={r.routing_reason} />
          <Row label="Model" value={`${reply.model} · ${reply.engine}`} />
        </Section>

        <Section title="Clasificator domeniu" note={r.classifier_input ? classifierInputLabel[r.classifier_input] ?? r.classifier_input : undefined}>
          {top.length > 0 ? (
            <>
              {top.map(([domain, p]) => (
                <Bar key={domain} label={domain} value={p} highlight={domain === r.effective_domain} />
              ))}
              {rest > 0 && <Bar label="altele" value={rest} />}
              {r.inherited && (
                <p className="note">
                  Decizia nu urmează clasificatorul: domeniul și complexitatea sunt moștenite din tura anterioară.
                </p>
              )}
            </>
          ) : (
            <p className="note">Clasificatorul nu a rulat pentru acest mesaj: decizia vine din conversație.</p>
          )}
        </Section>

        <Section title="Complexitate">
          {r.complexity_score != null ? (
            <Bar label="scor" value={r.complexity_score} highlight />
          ) : (
            <p className="note">Clasificatorul de complexitate nu a rulat pentru acest mesaj.</p>
          )}
          {r.inherited && r.effective_complexity != null && <Bar label="moștenit" value={r.effective_complexity} />}
        </Section>

        <Section title="Context (RAG)" note={r.rag_variant}>
          <Row label="Ture recente întregi" value={fmtInt(r.recent_turns)} />
          <Row label="Fragmente în context" value={ret?.candidates != null ? `${ids.length} din ${fmtInt(ret.candidates)} candidați` : fmtInt(ids.length)} />
          <Row label="Tokeni de context (estimat)" value={fmtInt(r.context_tokens)} />
          <Row label="Tokeni intrare / ieșire" value={`${fmtInt(reply.prompt_tokens)} / ${fmtInt(reply.completion_tokens)}`} />
          {ret?.error && <p className="note note-warning">Retrieval eșuat, doar turele recente: {ret.error}</p>}
          {ids.length > 0 && (
            <div className="ids">
              {ids.map((id) => (
                <span key={id} className="chip">
                  #{id}
                </span>
              ))}
            </div>
          )}
        </Section>

        <Section title="Timpi">
          <Row label="Clasificare" value={fmtMs(r.classify_ms)} />
          {ret && !ret.error && (
            <Row
              label="Retrieval (embed · search · rank)"
              value={`${fmtMs(ret.embed_ms)} · ${fmtMs(ret.search_ms)} · ${fmtMs(ret.rank_ms)}`}
            />
          )}
          <Row label="Generare" value={fmtMs(reply.latency_ms)} />
          <Row label="Total (în browser)" value={fmtMs(clientMs)} />
        </Section>
      </div>
    </aside>
  );
}
