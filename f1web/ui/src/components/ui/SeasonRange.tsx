import { getStatus } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import './JobOptions.css'

/** Season range for a pipeline run (CLI --start/--end); null = config default. */
export interface SeasonRangeValue {
  start: number | null
  end: number | null
}

export const DEFAULT_SEASON_RANGE: SeasonRangeValue = { start: null, end: null }

/** Payload fragment: only include seasons the user actually set. */
export function seasonPayload(range: SeasonRangeValue): Record<string, number> {
  return {
    ...(range.start !== null ? { start: range.start } : {}),
    ...(range.end !== null ? { end: range.end } : {}),
  }
}

/**
 * Resolve blank inputs to the allowed window (floor–ceiling) so a pipeline
 * run with empty seasons uses the same range the pickers allow — matching
 * the hint. Returns the raw range unchanged while status hasn't loaded yet.
 * The Data tab does NOT use this: fetching new seasons keeps the configured
 * range (it widens the window to it).
 */
export function resolveRange(
  range: SeasonRangeValue,
  seasons: { data_start: number; data_end: number } | null,
): SeasonRangeValue {
  if (!seasons) return range
  return {
    start: range.start ?? seasons.data_start,
    end: range.end ?? seasons.data_end,
  }
}

/**
 * Season-range inputs (CLI --start/--end). Empty inputs resolve to the
 * allowed window at job-submit time (see `resolveRange`), so a blank run uses
 * the same range the pickers allow; placeholders show the configured defaults
 * (`[data] start_season` / `end_season`) once /api/status is loaded.
 *
 * The allowed window is clamped: the start floor is the modern era (2014,
 * `seasons.data_start`) and the end ceiling is the latest season with fetched
 * data (`seasons.data_end`), so a pipeline run never silently references
 * seasons that have no data. Pages that fetch *new* seasons (the Data tab)
 * pass explicit `min`/`max` to widen the window (down to the configured start,
 * up to the configured end).
 */
export function SeasonRange({
  value,
  onChange,
  min,
  max,
}: {
  value: SeasonRangeValue
  onChange: (value: SeasonRangeValue) => void
  /** Override the start floor (default: modern era, `seasons.data_start`). */
  min?: number
  /** Override the end ceiling (default: latest season with fetched data). */
  max?: number
}) {
  const { state } = useApi('season-range-meta', () => getStatus())
  const seasons = state.phase === 'ready' ? state.data.seasons : null
  const floor = min ?? seasons?.data_start ?? 2014
  const ceiling = max ?? seasons?.data_end ?? seasons?.end
  const set = (field: 'start' | 'end', text: string) =>
    onChange({ ...value, [field]: text === '' ? null : Number(text) })
  const clamp = (field: 'start' | 'end') => () => {
    const current = value[field]
    if (current === null) return
    const lower = field === 'start' ? floor : seasons?.start
    const upper = ceiling
    if (current < lower || current > upper) {
      onChange({ ...value, [field]: Math.min(Math.max(current, lower), upper) })
    }
  }
  return (
    <>
      <div className="job-option">
        <label className="job-label" htmlFor="job-start-season">
          Start season
        </label>
        <input
          id="job-start-season"
          type="number"
          min={floor}
          max={ceiling}
          placeholder={seasons ? String(seasons.start) : 'start'}
          value={value.start ?? ''}
          onChange={(e) => set('start', e.target.value)}
          onBlur={clamp('start')}
        />
      </div>
      <div className="job-option">
        <label className="job-label" htmlFor="job-end-season">
          End season
        </label>
        <input
          id="job-end-season"
          type="number"
          min={seasons?.start}
          max={ceiling}
          placeholder={seasons ? String(seasons.end) : 'end'}
          value={value.end ?? ''}
          onChange={(e) => set('end', e.target.value)}
          onBlur={clamp('end')}
        />
      </div>
      {seasons ? (
        <p className="job-option-hint">
          Allowed window: {floor}–{ceiling}; blank uses this range.
        </p>
      ) : null}
    </>
  )
}
