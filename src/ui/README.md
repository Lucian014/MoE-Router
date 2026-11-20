# ui

The web interface: chat, conversation list and the router's decision for every answer (domain,
complexity, model, inherited, RAG fragments, timings). React + TypeScript + Vite.

- Calls only `/api/...` with relative paths (`src/api.ts`): Vite proxies them to the cluster in
  development; in production the gateway serves both the UI and the API.
- Conversations are kept in the browser (`localStorage`, `src/conversations.ts`) until the gateway
  has `GET /api/conversations`.
- Colors match the Figma file (`src/index.css`, light and dark).

```bash
npm install
npm run dev      # http://localhost:5173, /api -> http://llm-router (vite.config.ts)
npm run lint
npm run build    # tsc + dist/
```

| file | what it is |
|---|---|
| `src/App.tsx` | state: conversations, the pending request, the open details panel |
| `src/api.ts` | `POST /api/generate` and the reply's types |
| `src/format.ts` | Romanian labels for the router's codes, number formats |
| `src/components/` | `Sidebar`, `Messages` (answers, routing chips, errors), `Composer`, `DetailsPanel`, `EmptyState`, `icons` |
