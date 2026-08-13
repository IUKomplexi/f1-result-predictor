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
 */
export function SeasonRange({
  value,
  onChange,
}: {
  value: SeasonRangeValue
  onChange: (value: SeasonRangeValue) => void
}) {
  const { state } = useApi('season-range-meta', () => getConfig())
  const seasons = state.phase === 'ready' ? state.data.seasons : null
  const set = (field: 'start' | 'end', text: string) =>
    onChange({ ...value, [field]: text === '' ? null : Number(text) })
  return (
    <>
      <div className="job-option">
        <label className="job-label" htmlFor="job-start-season">
          Start season
        </label>
        <input
          id="job-start-season"
          type="number"
          min={seasons?.min}
          max={seasons?.max}
          placeholder={seasons ? String(seasons.min) : 'start'}
          value={value.start ?? ''}
          onChange={(e) => set('start', e.target.value)}
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
          max={seasons?.max}
          placeholder={seasons ? String(seasons.max) : 'end'}
          value={value.end ?? ''}
          onChange={(e) => set('end', e.target.value)}
        />
      </div>
    </>
  )
}
