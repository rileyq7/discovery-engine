import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return v.toFixed(digits);
}

export function fmtHours(h: number | null | undefined): string {
  if (h === null || h === undefined) return "–";
  if (h < 1 / 60) return `${Math.round(h * 3600)}s`;
  if (h < 1) return `${(h * 60).toFixed(1)} min`;
  return `${h.toFixed(2)} h`;
}

export function fmtTime(iso: string | null | undefined): string {
  if (!iso) return "–";
  const d = new Date(iso);
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  return sameDay
    ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString([], { month: "short", day: "numeric" }) + " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function shortId(id: string): string {
  // exp-20261011-192140-5e1d -> exp…192140-5e1d
  const m = id.match(/^(\w+)-\d{8}-(\d{6}-[0-9a-f]{4})$/);
  return m ? `${m[1]}·${m[2]}` : id;
}

export function navigate(hash: string) {
  window.location.hash = hash;
}
