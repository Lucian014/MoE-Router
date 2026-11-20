// The conversation list, kept in the browser until the gateway has GET /api/conversations.
import type { GenerateResponse } from "./api";

export type Message =
  | { id: string; role: "user"; text: string }
  | { id: string; role: "assistant"; reply: GenerateResponse; clientMs: number }
  | { id: string; role: "error"; text: string; prompt: string };

export type AssistantMessage = Extract<Message, { role: "assistant" }>;

export interface Conversation {
  id: string; // local key
  sessionId: string | null; // the gateway's, known after the first answer
  title: string;
  updatedAt: number;
  messages: Message[];
}

const KEY = "llm-router.conversations.v1";

export function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Conversation[]) : [];
  } catch {
    return [];
  }
}

export function saveConversations(list: Conversation[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(list));
  } catch {
    // storage full or blocked: the open conversation still works, only the list is lost on reload
  }
}

/** crypto.randomUUID() exists only in secure contexts, and http://llm-router is not one. */
export function newId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
}

export function titleFrom(prompt: string): string {
  const line = prompt.trim().split("\n")[0];
  return line.length > 48 ? line.slice(0, 47).trimEnd() + "…" : line;
}

/** Sidebar groups, newest first. */
export function groupByDay(list: Conversation[], now = new Date()): [string, Conversation[]][] {
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const day = 24 * 60 * 60 * 1000;
  const groups: [string, number][] = [
    ["Azi", startOfToday],
    ["Ieri", startOfToday - day],
    ["Ultimele 7 zile", startOfToday - 7 * day],
    ["Mai vechi", -Infinity],
  ];
  const out = new Map<string, Conversation[]>();
  for (const c of [...list].sort((a, b) => b.updatedAt - a.updatedAt)) {
    const label = groups.find(([, from]) => c.updatedAt >= from)![0];
    out.set(label, [...(out.get(label) ?? []), c]);
  }
  return [...out.entries()];
}
