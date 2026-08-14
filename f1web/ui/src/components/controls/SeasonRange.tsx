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
 * Validation message when the range is inverted (start after end), else null.
 * Both pickers clamp to the same floor, but a typed start > end is still
 * possible; callers use this to block submission and show a clear message.
 */
export function seasonRangeError(range: SeasonRangeValue): string | null {
  if (range.start !== null && range.end !== null && range.start > range.end) {
    return `Start season (${range.start}) is after end season (${range.end}) — clear or swap one.`
  }
  return null
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
 * The allowed window is clamped to ``[floor, ceiling]`` for BOTH inputs: the
 * start floor is the modern era (2014, ``seasons.data_start``) and the end
 * ceiling is the latest season with fetched data (``seasons.data_end``), so a
 * pipeline run never silently references seasons that have no data — and the
 * end season can never dip below the start floor. Pages that fetch *new*
 * seasons (the Data tab) pass explicit ``min``/``max`` to widen the window
 * (down to the configured start, up to the configured end).
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
  const rangeError = seasonRangeError(value)
  const clamp = (field: 'start' | 'end') => () => {
    const current = value[field]
    if (current === null) return
    // Both inputs share the same floor (the modern era) so the end season can
    // never be set below the start floor (previously it used the configured
    // start, allowing e.g. end=2010 while start was clamped to 2014).
    if (current < floor || current > ceiling) {
      onChange({ ...value, [field]: Math.min(Math.max(current, floor), ceiling) })
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
          min={floor}
          max={ceiling}
          placeholder={seasons ? String(seasons.end) : 'end'}
          value={value.end ?? ''}
          onChange={(e) => set('end', e.target.value)}
          onBlur={clamp('end')}
        />
      </div>
      {rangeError ? (
        <p className="job-option-hint range-error" role="alert">
          {rangeError}
        </p>
      ) : null}
    </>
  )
}
