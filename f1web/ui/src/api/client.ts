/**
 * Typed client for the internal predictor API (endpoints in f1web/app.py).
 *
 * All requests are same-origin: the Vite dev server proxies `/api` to the
 * FastAPI backend on :8080, and in production FastAPI serves the built SPA from
 * the same origin. Every error response has the shape `{"error": string}`,
 * surfaced here as an `ApiError` with the HTTP status.
 */

export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/* ------------------------------------------------------------------ types */

export interface RaceMeta {
  race_name: string | null
  circuit_id: string | null
  date: string | null
}

export interface PredictionRow {
  pred_rank: number
  driver_id: string
  constructor_id: string
  grid: number | null
  expected_points: number
  p_scored: number
  p_top3: number
  p_win: number
  actual_points: number | null
  actual_position: number | null
}

export interface Prediction {
  season: number
  round: number
  race: RaceMeta
  synthetic: boolean
  verified: boolean
  calibrated: boolean
  checkpoint: string
  drivers: PredictionRow[]
}

export interface BacktestMetricRow {
  winner_hit: number
  top3_overlap: number
  spearman: number
  mae: number
}

export interface Backtest {
  overall: Record<string, BacktestMetricRow>
  by_season: Record<string, Record<string, BacktestMetricRow>>
  /** Per-checkpoint comparison results (present when run with model_paths). */
  models?: Record<string, Pick<Backtest, 'overall' | 'by_season'>>
}

export interface CalendarEntry {
  season: number
  round: number
  race_name: string
  date: string
  time: string | null
  circuit_id: string | null
  circuit_name: string | null
  country: string | null
  circuit_lat: string | null
  circuit_long: string | null
  is_sprint_round: boolean
}

export interface Calendar {
  season: number
  calendar: CalendarEntry[]
}

export interface Status {
  seasons: {
    start: number
    end: number
    /** Dashboard clamp floor for pipeline season pickers (modern era). */
    data_start: number
    /** Latest season with fetched raw data (pipeline picker ceiling). */
    data_end: number
  }
  model: {
    checkpoint: string
    calibrators: string
    has_checkpoint: boolean
    has_calibrators: boolean
  }
  data: {
    dataset: string
    has_dataset: boolean
    has_raw_cache: boolean
  }
  reports: {
    has_backtest: boolean
    has_calibration: boolean
  }
  dashboard: { built: boolean }
}

export interface ModelInfo {
  checkpoint: string
  params?: Record<string, number>
  features?: string[]
  fingerprint?: string
  season_range?: [number, number]
  rows?: number
  seasons?: number
  trained_at?: number
}

export interface ModelsResponse {
  models: Record<string, ModelInfo>
  default: string
}

/* --------------------------------------------------------------- request */

async function apiJson<T>(
  method: 'GET' | 'POST' | 'PUT',
  path: string,
  body?: unknown,
): Promise<T> {
  const resp = await fetch(path, {
    method,
    headers: {
      Accept: 'application/json',
      ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  const respBody: unknown = await resp.json().catch(() => null)
  if (!resp.ok) {
    const error = (respBody as { error?: unknown } | null)?.error
    const message =
      typeof error === 'string' ? error : `Request failed with status ${resp.status}`
    throw new ApiError(message, resp.status)
  }
  return respBody as T
}

async function apiGet<T>(path: string): Promise<T> {
  return apiJson<T>('GET', path)
}

function qs(params: Record<string, string | number | boolean | null | undefined>): string {
  const entries = Object.entries(params).filter(
    (entry): entry is [string, string | number | boolean] =>
      entry[1] !== null && entry[1] !== undefined,
  )
  const search = new URLSearchParams(entries.map(([k, v]) => [k, String(v)])).toString()
  return search ? `?${search}` : ''
}

/* -------------------------------------------------------------- endpoints */

export function getStatus(): Promise<Status> {
  return apiGet<Status>('/api/status')
}

export function getModels(): Promise<ModelsResponse> {
  return apiGet<ModelsResponse>('/api/models')
}

export function getPrediction(season?: number, round?: number, refresh = false): Promise<Prediction> {
  return apiGet<Prediction>(`/api/prediction${qs({ season, round, refresh: refresh || undefined })}`)
}

export interface SeasonPredictions {
  season: number
  predictions: Prediction[]
}

/** All completed rounds of a season in one dataset pass (Race History). */
export function getSeasonPredictions(season: number): Promise<SeasonPredictions> {
  return apiGet<SeasonPredictions>(`/api/predictions/season${qs({ season })}`)
}

export function getBacktest(): Promise<Backtest> {
  return apiGet<Backtest>('/api/backtest')
}

export function getCalendar(season: number): Promise<Calendar> {
  return apiGet<Calendar>(`/api/calendar${qs({ season })}`)
}

/* ------------------------------------------------- config / jobs / predict */

/** One editable field in the config schema (see f1core.config.SCHEMA). */
export interface ConfigField {
  section: string
  key: string
  type: 'str' | 'int' | 'float' | 'bool' | 'list[str]' | 'params' | 'features'
  min?: number
  max?: number
  help?: string
}

export interface ConfigResponse {
  config: Record<string, Record<string, unknown>>
  schema: ConfigField[]
  features: {
    registry: string[]
    defaults: string[]
    categories: Record<string, string>
    /** Display order + labels for the feature groups (see features/registry.py). */
    category_meta: { id: string; label: string }[]
  }
  seasons: {
    min: number
    max: number
    /** Dashboard clamp floor for pipeline season pickers (modern era). */
    data_start: number
    /** Latest season with fetched raw data (pipeline picker ceiling). */
    data_end: number
  }
  model_params_keys: string[]
  jobs: string[]
}

export interface JobSummary {
  id: string
  type: string
  label: string
  status: 'queued' | 'running' | 'done' | 'failed'
  error: string | null
  created_at: number
  started_at: number | null
  finished_at: number | null
  /** Seconds since started_at (running) or total duration (finished). */
  elapsed_s: number | null
  /** Number of log lines; fetch the full log via getJob(id). */
  log_lines: number
}

export interface Job extends JobSummary {
  payload: Record<string, unknown>
  log: string[]
  result: Record<string, unknown> | null
}

export interface PredictOverrides {
  season?: number
  round?: number
  grid_csv?: string
  enable_features?: string[]
  disable_features?: string[]
  /** Ignore the raw-data cache (CLI --refresh). */
  refresh?: boolean
  /** Model checkpoint override (CLI --model). */
  model_path?: string
  /** Also write the Markdown report the CLI produces (CLI --out). */
  write_report?: boolean
}

export function getConfig(): Promise<ConfigResponse> {
  return apiGet<ConfigResponse>('/api/config')
}

export function putConfig(cfg: Record<string, Record<string, unknown>>): Promise<ConfigResponse> {
  return apiJson<ConfigResponse>('PUT', '/api/config', cfg)
}

export function postJob(type: string, payload?: Record<string, unknown>): Promise<{ id: string }> {
  return apiJson<{ id: string }>('POST', '/api/jobs', { type, payload: payload ?? {} })
}

export function getJobs(): Promise<{ jobs: JobSummary[] }> {
  return apiGet<{ jobs: JobSummary[] }>('/api/jobs')
}

export function getJob(id: string): Promise<Job> {
  return apiGet<Job>(`/api/jobs/${id}`)
}

export function postPrediction(overrides: PredictOverrides): Promise<Prediction> {
  return apiJson<Prediction>('POST', '/api/predict', overrides)
}
