import type { BacktestMetricRow } from '../../api/client'

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
