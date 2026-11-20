import { useMemo, useState } from "react";
import { groupByDay, type Conversation } from "../conversations";
import { MoonIcon, PanelIcon, PlusIcon, RouteIcon, SearchIcon, SunIcon } from "./icons";

export function Sidebar({
  conversations,
  activeId,
  onSelect,
  onNew,
  onCollapse,
  theme,
  onToggleTheme,
}: {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onCollapse: () => void;
  theme: "light" | "dark";
  onToggleTheme: () => void;
}) {
  const [query, setQuery] = useState("");
  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    return groupByDay(q ? conversations.filter((c) => c.title.toLowerCase().includes(q)) : conversations);
  }, [conversations, query]);

  return (
    <nav className="sidebar" aria-label="Conversații">
      <div className="brand">
        <span className="logo">
          <RouteIcon size={18} />
        </span>
        <span className="brand-name">LLM-Router</span>
        <button className="icon-button" onClick={onCollapse} aria-label="Ascunde bara laterală">
          <PanelIcon size={20} />
        </button>
      </div>

      <button className={`side-row side-new${activeId === null ? " is-active" : ""}`} onClick={onNew}>
        <PlusIcon /> Conversație nouă
      </button>
      <label className="side-row side-search">
        <SearchIcon />
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Caută conversații" />
      </label>

      <div className="side-list">
        {groups.map(([label, items]) => (
          <div key={label}>
            <div className="side-group">{label}</div>
            {items.map((c) => (
              <button
                key={c.id}
                className={`side-item${c.id === activeId ? " is-active" : ""}`}
                onClick={() => onSelect(c.id)}
                title={c.title}
              >
                {c.title}
              </button>
            ))}
          </div>
        ))}
        {conversations.length === 0 && <p className="side-empty">Conversațiile tale apar aici.</p>}
      </div>

      <div className="side-footer">
        <span className="muted">Conversațiile sunt salvate în acest browser.</span>
        <button className="icon-button" onClick={onToggleTheme} aria-label="Schimbă tema">
          {theme === "dark" ? <SunIcon /> : <MoonIcon />}
        </button>
      </div>
    </nav>
  );
}
