import type { ModelsResponse, PredictOverrides } from '../../api/client'
import { DEFAULT_MODEL } from '../backtest/lib'
import type { FeatureOverride } from '../ui/FeatureToggles'

/** The "config default model" pseudo-choice (no model_path is sent). */
export const RACE_DEFAULT_MODEL = DEFAULT_MODEL

/** Everything the Race tab can override for a single prediction request. */
export interface RaceOverrides {
  season: number
  round: number
  /** Ignore the raw-data cache (CLI --refresh). */
  refresh: boolean
  /** Model choice: a saved model name or RACE_DEFAULT_MODEL. */
  model: string
  /** Per-request feature toggles (CLI --enable/--disable-features). */
  features: FeatureOverride
  /** Editable grid rows (driver_id -> position text); null = no override. */
  gridRows: Record<string, string> | null
  /** Also write reports/prediction.md (CLI --out). */
  writeReport: boolean
}

export interface GridCheck {
  /** The serialized driver_id,grid CSV, or null when no cell has a value. */
  csv: string | null
  /** User-facing validation message; null when the grid is valid. */
  error: string | null
}

/**
 * Validate editable grid rows and serialize them to the driver_id,grid CSV
 * the backend expects. Empty cells are skipped (the model's own grid stays
 * for those drivers); every non-empty cell must be a positive integer.
 * ``csv`` is null when no cell has a value (no override to send).
 */
export function checkGrid(rows: Record<string, string>): GridCheck {
  const entries: string[] = []
  for (const [driverId, position] of Object.entries(rows)) {
    const trimmed = position.trim()
    if (trimmed === '') continue
    if (!/^\d+$/.test(trimmed)) {
      return {
        csv: null,
        error: `grid position for ${driverId} must be a positive integer`,
      }
    }
    entries.push(`${driverId},${trimmed}`)
  }
  return {
    csv: entries.length > 0 ? `driver_id,grid\n${entries.join('\n')}` : null,
    error: null,
  }
}

/** The checkpoint path a model choice maps to; undefined for the config default. */
export function modelPathFor(models: ModelsResponse | null, choice: string): string | undefined {
  if (!models || choice === RACE_DEFAULT_MODEL) return undefined
  return models.models[choice]?.checkpoint
}

/**
 * Build the POST /api/predict body for the current Race-tab overrides
 * (mirrors the CLI's predict flags). ``error`` carries a user-facing message
 * when the grid is invalid; the body is then not meant to be sent.
 */
export function racePredictBody(
  models: ModelsResponse | null,
  overrides: RaceOverrides,
): { body: PredictOverrides; error: string | null } {
  const body: PredictOverrides = {
    season: overrides.season,
    round: overrides.round,
    refresh: overrides.refresh || undefined,
    model_path: modelPathFor(models, overrides.model),
    write_report: overrides.writeReport || undefined,
    enable_features: overrides.features.enable.length > 0 ? overrides.features.enable : undefined,
    disable_features: overrides.features.disable.length > 0 ? overrides.features.disable : undefined,
  }
  if (overrides.gridRows !== null) {
    const check = checkGrid(overrides.gridRows)
    if (check.error !== null) return { body, error: check.error }
    if (check.csv !== null) body.grid_csv = check.csv
  }
  return { body, error: null }
}
