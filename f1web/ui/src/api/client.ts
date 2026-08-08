/**
 * Typed client for the internal predictor API (endpoints in f1web/app.py).
 *
 * All requests are same-origin: the Vite dev server proxies `/api` to the
 * Flask backend on :8080, and in production Flask serves the built SPA from
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
}

export interface ReliabilityBin {
  mean_pred: number
  observed: number
  n: number
}

export interface CalibrationTarget {
  brier_raw: number
  brier_calibrated: number
  delta: number
  deployed: boolean
  reliability: ReliabilityBin[]
}

export interface Calibration {
  context: string
  deployed: string[]
  targets: Record<string, CalibrationTarget>
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

export interface StandingRow {
  season: number
  round: number | null
  position: number | null
  points: number
  wins: number | null
  driver_id?: string | null
  constructor_id?: string | null
}

export interface Standings {
  season: number
  round: number | null
  driver: StandingRow[]
  constructor: StandingRow[]
}

export interface Status {
  seasons: { start: number; end: number }
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

/* --------------------------------------------------------------- request */

async function apiGet<T>(path: string): Promise<T> {
  const resp = await fetch(path, { headers: { Accept: 'application/json' } })
  const body: unknown = await resp.json().catch(() => null)
  if (!resp.ok) {
    const error = (body as { error?: unknown } | null)?.error
    const message =
      typeof error === 'string' ? error : `Request failed with status ${resp.status}`
    throw new ApiError(message, resp.status)
  }
  return body as T
}

function qs(params: Record<string, string | number | null | undefined>): string {
  const entries = Object.entries(params).filter(
    (entry): entry is [string, string | number] =>
      entry[1] !== null && entry[1] !== undefined,
  )
  const search = new URLSearchParams(entries.map(([k, v]) => [k, String(v)])).toString()
  return search ? `?${search}` : ''
}

/* -------------------------------------------------------------- endpoints */

export function getStatus(): Promise<Status> {
  return apiGet<Status>('/api/status')
}

export function getPrediction(season?: number, round?: number): Promise<Prediction> {
  return apiGet<Prediction>(`/api/prediction${qs({ season, round })}`)
}

export function getBacktest(): Promise<Backtest> {
  return apiGet<Backtest>('/api/backtest')
}

export function getCalibration(): Promise<Calibration> {
  return apiGet<Calibration>('/api/calibration')
}

export function getCalendar(season: number): Promise<Calendar> {
  return apiGet<Calendar>(`/api/calendar${qs({ season })}`)
}

export function getStandings(season: number, round?: number): Promise<Standings> {
  return apiGet<Standings>(`/api/standings${qs({ season, round })}`)
}
