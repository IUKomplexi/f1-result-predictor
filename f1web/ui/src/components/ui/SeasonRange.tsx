import { getConfig } from '../../api/client'
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
 * Season-range inputs (CLI --start/--end). Empty inputs fall back to the
 * config season range at job-submit time (see f1web/jobs._cfg_start_end);
 * placeholders show the configured defaults once /api/config is loaded.
 *
 * The allowed window is clamped: the start floor is the modern era (2014,
 * `seasons.data_start`) and the end ceiling is the latest season with fetched
 * data (`seasons.data_end`), so a pipeline run never silently references
 * seasons that have no data. Pages that fetch *new* seasons (the Data tab)
 * pass explicit `min`/`max` to widen the window.
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
  const { state } = useApi('season-range-meta', () => getConfig())
  const seasons = state.phase === 'ready' ? state.data.seasons : null
  const floor = min ?? seasons?.data_start ?? 2014
  const ceiling = max ?? seasons?.data_end ?? seasons?.max
  const set = (field: 'start' | 'end', text: string) =>
    onChange({ ...value, [field]: text === '' ? null : Number(text) })
  const clamp = (field: 'start' | 'end') => () => {
    const current = value[field]
    if (current === null) return
    const lower = field === 'start' ? floor : seasons?.min
    const upper = field === 'start' ? ceiling : ceiling
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
          placeholder={seasons ? String(seasons.min) : 'start'}
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
          min={seasons?.min}
          max={ceiling}
          placeholder={seasons ? String(seasons.max) : 'end'}
          value={value.end ?? ''}
          onChange={(e) => set('end', e.target.value)}
          onBlur={clamp('end')}
        />
      </div>
      {seasons ? (
        <p className="job-option-hint">
          Allowed window: {floor}–{ceiling} (modern era up to the latest season
          with fetched data).
        </p>
      ) : null}
    </>
  )
}
