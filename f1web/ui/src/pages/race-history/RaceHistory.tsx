import { useEffect, useRef, useState } from 'react'
import {
  ApiError,
  getModels,
  getSeasonPredictions,
  getStatus,
  type Job,
  type ModelsResponse,
} from '../../api/client'
import type { TabProps } from '../../types'
import { useApi } from '../../hooks/useApi'
import { analyzeRace, type RaceResult } from '../../lib/analysis'
import { driverLabel, fmtDate } from '../../lib/format'
import { Badge } from '../../components/ui/Badge'
import { ErrorState, Skeleton } from '../../components/ui/DataState'
import { JobRunner } from '../../components/jobs/JobRunner'
import { ModelPicker } from '../race/ModelPicker'
import { RACE_DEFAULT_MODEL, modelPathFor } from '../race/lib'
import {
  DEFAULT_SEASON_RANGE,
  SeasonRange,
  resolveRange,
  seasonPayload,
  seasonRangeError,
  type SeasonRangeValue,
} from '../../components/controls/SeasonRange'
import '../race/Race.css'
import './RaceHistory.css'

type SeasonState =
  | { phase: 'idle' }
  | { phase: 'loading'; label: string; done: number; total: number }
  | { phase: 'error'; message: string }
  | { phase: 'ready'; races: RaceResult[] }

function useSeasonResults(
  season: number | null,
  modelPath: string | null,
): {
  state: SeasonState
  retry: () => void
} {
  const [attempt, setAttempt] = useState(0)
  const [state, setState] = useState<SeasonState>({ phase: 'idle' })

  useEffect(() => {
    if (season === null) {
      setState({ phase: 'idle' })
      return
    }
    let cancelled = false
    setState({ phase: 'loading', label: 'Fetching season predictions…', done: 0, total: 0 })
    ;(async () => {
      try {
        // One dataset pass for the whole season (backend /api/predictions/season),
        // instead of N sequential per-round recomputes.
        const data = await getSeasonPredictions(season, modelPath)
        const predictions = Array.isArray(data.predictions) ? data.predictions : []
        const races = predictions
          .map(analyzeRace)
          .sort((a, b) => a.round - b.round)
        if (!cancelled) setState({ phase: 'ready', races })
      } catch (error) {
        if (cancelled) return
        const message =
          error instanceof ApiError ? error.message : String(error)
        setState({ phase: 'error', message })
      }
    })()
    return () => {
      cancelled = true
    }
  }, [season, modelPath, attempt])

  return { state, retry: () => setAttempt((n) => n + 1) }
}

export function RaceHistory({ onNavigate }: TabProps) {
  const status = useApi('status', () => getStatus())
  const modelsState = useApi<ModelsResponse>('models', getModels)
  const [season, setSeason] = useState<number | null>(null)
  const [model, setModel] = useState<string>(RACE_DEFAULT_MODEL)
  const [range, setRange] = useState<SeasonRangeValue>(DEFAULT_SEASON_RANGE)
  const refresh = false
  const primed = useRef(false)
  const models = modelsState.state.phase === 'ready' ? modelsState.state.data : null
  const modelPath = modelPathFor(models, model)
  const { state, retry } = useSeasonResults(season, modelPath ?? null)

  const statusState = status.state
  const seasons = statusState.phase === 'ready' ? statusState.data.seasons : null
  // Default the precompute range to the most recent three seasons.
  useEffect(() => {
    if (primed.current || statusState.phase !== 'ready') return
    primed.current = true
    const { start, end } = statusState.data.seasons
    setRange({ start: Math.max(end - 2, start), end })
  }, [statusState])

  if (status.state.phase === 'loading') return <Skeleton rows={3} />
  if (status.state.phase === 'error') {
    return <ErrorState message={status.state.message} onRetry={status.retry} />
  }
  const { start, end } = status.state.data.seasons
  const selected = season ?? end

  // After a precompute finishes, show the newest season (or refresh the
  // current selection) so the instant cache hit is visible immediately.
  const handlePrecomputed = () => {
    if (season === null) {
      setSeason(end)
    } else {
      retry()
    }
  }

  return (
    <>
      <section className="card">
        <JobRunner
          type="history"
          runLabel="Precompute race history"
          layout="action-end"
          onDone={handlePrecomputed}
          buildPayload={() => {
            const resolved = resolveRange(range, seasons)
            const rangeError = seasonRangeError(resolved)
            if (rangeError) throw new Error(rangeError)
            return { ...seasonPayload(resolved), refresh }
          }}
          options={
            <>
              <SeasonRange value={range} onChange={setRange} />
            </>
          }
          renderResult={(job) => <HistoryRunResult job={job} />}
        />
      </section>

      {state.phase === 'idle' || state.phase === 'loading' ? (
        <Skeleton rows={6} />
      ) : null}
      {state.phase === 'error' ? (
        <ErrorState message={state.message} onRetry={retry} />
      ) : null}
      {state.phase === 'ready' ? (
        <RaceHistoryTable
          races={state.races}
          models={models}
          selectedModel={model}
          selectedSeason={selected}
          startSeason={start}
          endSeason={end}
          onSeasonChange={setSeason}
          onModelChange={setModel}
          onOpenRace={(round) => onNavigate?.('race', { season: selected, round })}
        />
      ) : null}
    </>
  )
}

function HistoryRunResult({ job }: { job: Job }) {
  const result = job.result ?? {}
  const seasons = (result.seasons ?? {}) as Record<string, { rounds: number; elapsed_s: number }>
  return (
    <div className="result-block">
      <h3 className="card-title">Precomputed seasons</h3>
      {job.log.length > 0 && (
        <details className="job-log">
          <summary>Log</summary>
          {job.log.map((line, i) => (
            <pre key={i} className="log-line">{line}</pre>
          ))}
        </details>
      )}
      <ul className="summary-list">
        {Object.entries(seasons).map(([s, v]) => (
          <li key={s}>
            season {s}: <strong>{v.rounds}</strong> rounds ({v.elapsed_s}s)
          </li>
        ))}
      </ul>
    </div>
  )
}

function RaceHistoryTable({
  races,
  models,
  selectedModel,
  selectedSeason,
  startSeason,
  endSeason,
  onSeasonChange,
  onModelChange,
  onOpenRace,
}: {
  races: RaceResult[]
  models: ModelsResponse | null
  selectedModel: string
  selectedSeason: number
  startSeason: number
  endSeason: number
  onSeasonChange: (season: number) => void
  onModelChange: (model: string) => void
  onOpenRace: (round: number) => void
}) {
  const raced = races.filter((race) => race.hasActuals)
  const hits = raced.filter((race) => race.winnerHit).length
  const avgOverlap =
    raced.length > 0
      ? raced.reduce((sum, race) => sum + (race.top3Overlap ?? 0), 0) / raced.length
      : null

  return (
    <>
      <section className="card">
        <div className="race-nav history-selector-row">
          <label className="field">
            <span className="field-label">Season</span>
            <select
              className="select"
              value={selectedSeason}
              onChange={(event) => onSeasonChange(Number(event.target.value))}
            >
              {Array.from({ length: endSeason - startSeason + 1 }, (_, i) => endSeason - i).map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <ModelPicker models={models} value={selectedModel} onChange={onModelChange} />
        </div>
        <div className="metrics">
          <div className="metric">
            <span className="metric-label">Races with actuals</span>
            <span className="metric-value">
              {raced.length}/{races.length}
            </span>
            <span className="metric-sub">completed rounds this season</span>
          </div>
          <div className="metric">
            <span className="metric-label">Winner hits</span>
            <span className="metric-value">
              {hits}/{raced.length}
            </span>
            <span className="metric-sub">predicted winner = actual</span>
          </div>
          <div className="metric">
            <span className="metric-label">Avg top-3 overlap</span>
            <span className="metric-value">
              {avgOverlap !== null ? avgOverlap.toFixed(2) : '–'}
            </span>
            <span className="metric-sub">predicted ∩ actual podium</span>
          </div>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col" className="num">Rd</th>
                <th scope="col">Race</th>
                <th scope="col" className="hide-narrow">Date</th>
                <th scope="col">Predicted winner</th>
                <th scope="col">Actual winner</th>
                <th scope="col" className="num">Top-3 overlap</th>
                <th scope="col" className="num">Winner</th>
              </tr>
            </thead>
            <tbody>
              {races.map((race) => (
                <tr key={race.round}>
                  <td className="num">{race.round}</td>
                  <td>
                    <button
                      type="button"
                      className="link-button race-link"
                      title={`Open the round ${race.round} prediction in the Race tab`}
                      onClick={() => onOpenRace(race.round)}
                    >
                      {race.raceName}
                    </button>
                  </td>
                  <td className="muted hide-narrow">{fmtDate(race.date)}</td>
                  <td>{driverLabel(race.predictedWinner)}</td>
                  <td className={race.winnerHit === false ? 'miss-text' : undefined}>
                    {race.actualWinner ? driverLabel(race.actualWinner) : '–'}
                  </td>
                  <td className="num">
                    {race.top3Overlap !== null
                      ? `${race.top3Overlap}/${race.top3Of}`
                      : '–'}
                  </td>
                  <td className="num">
                    {race.winnerHit === null ? (
                      <span className="muted">–</span>
                    ) : race.winnerHit ? (
                      <Badge variant="ready">Hit</Badge>
                    ) : (
                      <Badge variant="missing">Miss</Badge>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  )
}
