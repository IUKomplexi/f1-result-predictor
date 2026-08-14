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
  { key: 'top10_overlap', label: 'Top-10 overlap' },
  { key: 'spearman', label: 'Spearman' },
  { key: 'mae', label: 'MAE' },
]

/** Plain-language meaning of the baselines (shown under the mean table). */
export const BASELINE_HELP: Record<string, string> = {
  grid: 'predicts the points of the grid slot',
  championship: 'predicts the points of the championship position',
  zero: 'predicts 0 points for every driver (the naive baseline)',
}

/** Metrics where a LOWER value is better (MAE; all others are higher-better). */
const LOWER_IS_BETTER: ReadonlySet<keyof BacktestMetricRow> = new Set(['mae'])

export type CellTone = 'best' | 'worst' | 'mid' | 'neutral'

/**
 * Tone for one cell in a metric column across several rows, so a comparison
 * table shows at a glance which value is best: best = green, worst = red,
 * strictly-between values = orange, ties/all-equal = white (no meaningful
 * difference).
 */
export function cellTone(
  metric: keyof BacktestMetricRow,
  values: number[],
  value: number,
): CellTone {
  const finite = values.filter((v) => Number.isFinite(v))
  if (finite.length === 0) return 'neutral'
  const best = LOWER_IS_BETTER.has(metric) ? Math.min(...finite) : Math.max(...finite)
  const worst = LOWER_IS_BETTER.has(metric) ? Math.max(...finite) : Math.min(...finite)
  if (best === worst) return 'neutral'
  if (value === best) return 'best'
  if (value === worst) return 'worst'
  return 'mid'
}

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
