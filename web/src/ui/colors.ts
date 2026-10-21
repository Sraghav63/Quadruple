// One colour per link, shared by canvas, HUD and plots (matches the CSS --l0..--l3 variables).
export const LINK_COLORS = ["#ff8a7a", "#8fe3b0", "#ffd27a", "#b9a2ff"];
export const linkColor = (i: number): string => LINK_COLORS[i % LINK_COLORS.length]!;

export const THEME = {
  bg: "#0d0d0d",
  rail: "#3a3a3a",
  stop: "#4d4d4d",
  target: "#78aaff",
  cart: "#e0e0e0",
  wheel: "#0d0d0d",
  fg: "#f0f0f0",
  dim: "#707070",
  grid: "rgba(255,255,255,0.035)",
  ok: "#4cc38a",
  bad: "#f2555a",
  warn: "#f0a94a",
} as const;
