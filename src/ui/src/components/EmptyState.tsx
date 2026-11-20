import { Composer } from "./Composer";
import { RouteIcon } from "./icons";

// Sample prompts for a first try; the domain is only a hint, the router decides.
const EXAMPLES: [string, string][] = [
  ["coding", "Scrie o funcție Python care validează un CNP."],
  ["math", "Rezolvă ecuația x² − 5x + 6 = 0 pas cu pas."],
  ["translation", "Tradu în engleză: „Routerul alege modelul potrivit pentru fiecare mesaj.”"],
  ["summarization", "Rezumă în 5 idei ce înseamnă Mixture of Experts."],
];

export function EmptyState({ onSend, busy }: { onSend: (text: string) => void; busy: boolean }) {
  return (
    <div className="empty">
      <span className="logo logo-lg">
        <RouteIcon size={28} />
      </span>
      <h1>Cu ce te pot ajuta?</h1>
      <p className="empty-sub">
        Fiecare mesaj ajunge automat la modelul potrivit: cod, matematică, traducere, rezumat, scriere creativă și
        altele.
      </p>
      <div className="column">
        <Composer onSend={onSend} busy={busy} autoFocus />
      </div>
      <div className="examples column">
        {EXAMPLES.map(([domain, prompt]) => (
          <button key={prompt} className="example" onClick={() => onSend(prompt)} disabled={busy}>
            <span className="chip">{domain}</span>
            <span>{prompt}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
