// One colour per link, shared by canvas, HUD and plots (matches the CSS --l0..--l3 variables).
export const LINK_COLORS = ["#eb5f4b", "#46be78", "#f0be3c", "#966edc"];
export const linkColor = (i: number): string => LINK_COLORS[i % LINK_COLORS.length]!;

export const THEME = {
  bg: "#0e1218",
  rail: "#6b7480",
  stop: "#8b95a3",
  target: "#5aa9ff",
  cart: "#c9d1da",
  wheel: "#161a21",
  fg: "#e6eaf0",
  dim: "#8a94a3",
  grid: "#1c232e",
  ok: "#46be78",
  bad: "#f0584a",
  warn: "#f0a03c",
} as const;
