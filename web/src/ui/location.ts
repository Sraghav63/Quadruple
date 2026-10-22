import type { Preset, PresetSummary } from '../sim/types';

export function parseHash(hash: string, presets: PresetSummary[]): { presetId: string; configuration?: string } {
  const [id, cfg, ...extra] = hash.replace(/^#/, '').split('/');
  const presetId = presets.some(p => p.id === id) ? id : presets.find(p => p.id === 'quad')?.id ?? presets[0].id;
  const configuration = extra.length === 0 && cfg && /^[UD]+$/i.test(cfg) ? cfg.toUpperCase() : undefined;
  return { presetId, configuration };
}

export function resolveConfiguration(preset: Preset, configuration?: string): string {
  return configuration && Object.hasOwn(preset.configurations, configuration) ? configuration : preset.controller.configuration;
}
