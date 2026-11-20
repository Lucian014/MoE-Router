// Every HTTP call of the UI. Paths are relative: in development Vite proxies /api to the cluster,
// in production the gateway serves both the UI and /api. (Authentication will be added here.)

export type ComplexityLevel = "high" | "low";

/** Retrieval of the request's RAG fragments (absent when the session had no history). */
export interface RetrievalInfo {
  candidates?: number;
  pool?: number;
  embed_ms?: number;
  search_ms?: number;
  rank_ms?: number;
  fetch_ms?: number;
  error?: string;
}

/** The router's decision, as the worker returns it (src/router/router.py). */
export interface Routing {
  routing_reason?: string; // match | other-level | fallback:no-model
  task_type?: string | null; // what the domain classifier said (null: it did not run)
  task_type_confidence?: number | null;
  task_type_probabilities?: Record<string, number> | null;
  complexity_score?: number | null; // what the complexity classifier said
  classifier_input?: string | null; // full | edges | cut
  session_routing?: string; // first-turn | classified | short-prompt | ambiguous-domain
  inherited?: boolean;
  effective_domain?: string; // what the request was routed on
  effective_domain_confidence?: number | null;
  effective_complexity?: number | null;
  complexity?: ComplexityLevel;
  classify_ms?: number;
  rag_variant?: string;
  recent_turns?: number;
  retrieved_chunk_ids?: number[];
  retrieval?: RetrievalInfo | null;
  context_tokens?: number;
}

export interface GenerateResponse {
  response: string;
  model: string;
  engine: string;
  latency_ms: number;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  truncated: boolean;
  turn: number | null;
  routing: Routing;
  session_id: string;
}

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/** POST /api/generate. Without a sessionId the gateway starts a new conversation. */
export async function generate(prompt: string, sessionId: string | null): Promise<GenerateResponse> {
  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(sessionId ? { prompt, session_id: sessionId } : { prompt }),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // not JSON: an error page from a proxy (Traefik, Vite)
    }
    throw new ApiError(res.status, detail || `HTTP ${res.status}`);
  }
  return res.json();
}
