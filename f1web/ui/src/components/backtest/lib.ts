import type { BacktestMetricRow, ModelsResponse } from '../../api/client'

export const BASELINES = ['model', 'grid', 'championship', 'zero'] as const
export type Baseline = (typeof BASELINES)[number]

export const BASELINE_LABEL: Record<Baseline, string> = {
  model: 'Model',
  grid: 'Grid',
  championship: 'Championship',
  zero: 'Zero',
}

export const BASELINE_COLOR: Record<Baseline, string> = {
  model: '#e10600',
  grid: '#4a7fd6',
  championship: '#d9a514',
  zero: '#6f6f7d',
}

/** Palette for per-model comparison series (baselines keep their own colors). */
export const MODEL_COLORS = ['#e10600', '#4a7fd6', '#d9a514', '#17a354', '#8b5cf6', '#0ea5e9', '#f97316']

export const METRICS: { key: keyof BacktestMetricRow; label: string }[] = [
  { key: 'winner_hit', label: 'Winner hit' },
  { key: 'top3_overlap', label: 'Top-3 overlap' },
  { key: 'spearman', label: 'Spearman' },
  { key: 'mae', label: 'MAE' },
]

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

// Metrics where a higher value is better. For the rest (MAE), lower is better,
// so the delta is computed as reference - model to keep "positive = better".
const IMPROVE_UP = new Set<keyof BacktestMetricRow>([
  'winner_hit',
  'top3_overlap',
  'spearman',
])

export function deltaVs(
  model: BacktestMetricRow | undefined,
  reference: BacktestMetricRow | undefined,
  metric: keyof BacktestMetricRow,
): number | null {
  const a = model?.[metric]
  const b = reference?.[metric]
  if (a === undefined || b === undefined) return null
  return IMPROVE_UP.has(metric) ? a - b : b - a
}

export function allSeasons(bySeason: Record<string, Record<string, BacktestMetricRow>>): number[] {
  const set = new Set<number>()
  for (const table of Object.values(bySeason)) {
    for (const season of Object.keys(table)) set.add(Number(season))
  }
  return [...set].sort((a, b) => a - b)
}
