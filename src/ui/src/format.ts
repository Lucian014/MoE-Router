// Display text for the router's codes and numbers, in Romanian.
import type { ComplexityLevel } from "./api";

const number = new Intl.NumberFormat("ro-RO");
const decimal = new Intl.NumberFormat("ro-RO", { minimumFractionDigits: 1, maximumFractionDigits: 1 });

export const fmtInt = (n: number | null | undefined) => (n == null ? "—" : number.format(n));
export const fmtMs = (ms: number | null | undefined) =>
  ms == null ? "—" : ms >= 1000 ? `${decimal.format(ms / 1000)} s` : `${number.format(Math.round(ms))} ms`;
export const fmtProb = (p: number | null | undefined) => (p == null ? "—" : p.toFixed(2));

export const complexityLabel = (c: ComplexityLevel | undefined) =>
  c === "high" ? "ridicată" : c === "low" ? "scăzută" : "—";

export const sessionRoutingLabel: Record<string, string> = {
  "first-turn": "prima tură: clasificat",
  classified: "clasificat",
  "short-prompt": "mesaj scurt: moștenit",
  "ambiguous-domain": "domeniu ambiguu: moștenit",
};

export const routingReasonLabel: Record<string, string> = {
  match: "model pentru domeniu și complexitate",
  "other-level": "alt nivel de complexitate",
  "fallback:no-model": "model implicit (niciun model pe domeniu)",
};

export const classifierInputLabel: Record<string, string> = {
  full: "mesajul întreg",
  edges: "cerere + începutul documentului",
  cut: "mesaj tăiat",
};

export const fragments = (n: number) => (n === 1 ? "1 fragment" : `${n} fragmente`);
