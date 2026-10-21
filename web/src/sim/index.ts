export type * from './types';
export { createSimulation, DEFAULT_KNOBS } from './simulation';
import type { Preset, PresetSummary } from './types';

function presetUrl(file: string, base: string): string {
  // Vite's base can be absolute, root-relative, or './'. Keep it intact and
  // resolve relative paths against the current document (also works in tests).
  return `${base.endsWith('/') ? base : `${base}/`}presets/${file}`;
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url}: HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

export function loadPresetIndex(base = import.meta.env.BASE_URL): Promise<PresetSummary[]> {
  return fetchJson(presetUrl('index.json', base));
}

export function loadPreset(id: string, base = import.meta.env.BASE_URL): Promise<Preset> {
  return fetchJson(presetUrl(`${encodeURIComponent(id)}.json`, base));
}
