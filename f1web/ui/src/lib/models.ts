import type { ModelsResponse } from '../api/client'

/**
 * Shared "which model is selected / deployed" helpers. Used by several tabs
 * (Race, Race History, Backtest, Settings), so they live here instead of in
 * any single tab's folder.
 */

/** The "deployed config model" pseudo-choice, only when it is not a saved model. */
export const DEFAULT_MODEL = 'default'

export interface ModelChoice {
  /** Selection value: a saved model name, or DEFAULT_MODEL for the config default. */
  value: string
  label: string
}

/** Normalize a checkpoint path for identity comparison (Windows separators). */
export function normPath(path: string | null | undefined): string | null {
  if (!path) return null
  return path.replace(/\\/g, '/').toLowerCase()
}

/** "max_iter=200 · learning_rate=0.05" pairs, keys sorted; null when absent. */
export function formatParams(params: Record<string, number> | undefined | null): string | null {
  if (!params) return null
  return Object.entries(params)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${k}=${v}`)
    .join(' · ')
}

/** The saved model whose checkpoint equals the config default, if any. */
export function deployedName(models: ModelsResponse | null): string | null {
  if (!models) return null
  const target = normPath(models.default)
  if (!target) return null
  for (const [name, info] of Object.entries(models.models)) {
    if (normPath(info.checkpoint) === target) return name
  }
  return null
}

export function modelChoices(models: ModelsResponse | null): ModelChoice[] {
  const saved = models ? Object.keys(models.models).sort() : []
  const deployed = deployedName(models)
  const choices: ModelChoice[] = saved.map((name) => ({
    value: name,
    label: deployed === name ? `${name} (deployed)` : name,
  }))
  if (!models || deployed !== null) return choices
  // The config default points at a checkpoint that is not in the saved index
  // (CLI-trained, or the [model] checkpoint was edited in Settings): keep a
  // pseudo-entry so that model stays selectable. When it IS a saved model we
  // skip this — that model is already listed once, marked (deployed).
  return [{ value: DEFAULT_MODEL, label: 'config default (deployed)' }, ...choices]
}

export function selectedPaths(models: ModelsResponse | null, checked: string[]): string[] {
  if (!models) return []
  const seen = new Set<string>()
  const paths: string[] = []
  for (const value of checked) {
    const path = value === DEFAULT_MODEL ? models.default : models.models[value]?.checkpoint
    if (typeof path !== 'string' || path.length === 0) continue
    const key = normPath(path)
    if (key === null || seen.has(key)) continue
    seen.add(key)
    paths.push(path)
  }
  return paths
}

export function defaultStem(models: ModelsResponse | null): string | null {
  if (!models) return null
  return models.default.split(/[\\/]/).pop()?.replace(/\.joblib$/, '') ?? null
}
