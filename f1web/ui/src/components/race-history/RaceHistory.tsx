import { useEffect, useState } from 'react'
import {
  ApiError,
  getCalendar,
  getPrediction,
  getStatus,
  type Prediction,
} from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { driverLabel, fmtDate } from '../../lib/format'
import { Badge } from '../ui/Badge'
import { ErrorState, ProgressState, Skeleton } from '../ui/DataState'
import './RaceHistory.css'

interface RaceResult {
  round: number
  raceName: string
  date: string
  predictedWinner: string
  actualWinner: string | null
  winnerHit: boolean | null
  top3Overlap: number | null
  top3Of: number
  hasActuals: boolean
}

function analyzeRace(prediction: Prediction): RaceResult {
  const sorted = [...prediction.drivers].sort((a, b) => a.pred_rank - b.pred_rank)
  const predictedTop3 = sorted.slice(0, 3).map((row) => row.driver_id)
  const predictedWinner = predictedTop3[0] ?? ''

  const raced = prediction.drivers.filter(
    (row) => row.actual_position !== null && row.actual_position !== undefined,
  )
  const winnerRow = raced.find((row) => row.actual_position === 1)
  if (!winnerRow) {
    return {
      round: prediction.round,
      raceName: prediction.race.race_name ?? `Round ${prediction.round}`,
      date: prediction.race.date ?? '',
      predictedWinner,
      actualWinner: null,
      winnerHit: null,
      top3Overlap: null,
      top3Of: 3,
      hasActuals: false,
    }
  }
  const actualByPosition = new Map(
    raced.map((row) => [row.actual_position as number, row.driver_id]),
  )
  const actualTop3 = [1, 2, 3]
    .map((pos) => actualByPosition.get(pos))
    .filter((id): id is string => id !== undefined)
  const overlap = predictedTop3.filter((id) => actualTop3.includes(id)).length
  return {
    round: prediction.round,
    raceName: prediction.race.race_name ?? `Round ${prediction.round}`,
    date: prediction.race.date ?? '',
    predictedWinner,
    actualWinner: actualByPosition.get(1) ?? null,
    winnerHit: actualByPosition.get(1) === predictedWinner,
    top3Overlap: overlap,
    top3Of: Math.min(3, actualTop3.length),
    hasActuals: true,
  }
}

type SeasonState =
  | { phase: 'idle' }
  | { phase: 'loading'; label: string; done: number; total: number }
  | { phase: 'error'; message: string }
  | { phase: 'ready'; races: RaceResult[] }

function useSeasonResults(season: number | null): {
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
    setState({ phase: 'loading', label: 'Fetching calendar…', done: 0, total: 0 })
    ;(async () => {
      try {
        const calendar = await getCalendar(season)
        const rounds = calendar.calendar.map((entry) => entry.round)
        setState({
          phase: 'loading',
          label: `Predicting 0/${rounds.length} races…`,
          done: 0,
          total: rounds.length,
        })
        const races: RaceResult[] = []
        for (const round of rounds) {
          if (cancelled) return
          const prediction = await getPrediction(season, round)
          races.push(analyzeRace(prediction))
          setState({
            phase: 'loading',
            label: `Predicting ${races.length}/${rounds.length} races…`,
            done: races.length,
            total: rounds.length,
          })
        }
        races.sort((a, b) => a.round - b.round)
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
  }, [season, attempt])

  return { state, retry: () => setAttempt((n) => n + 1) }
}

export function RaceHistory() {
  const status = useApi('status', () => getStatus())
  const [season, setSeason] = useState<number | null>(null)
  const { state, retry } = useSeasonResults(season)

  if (status.state.phase === 'loading') return <Skeleton rows={3} />
  if (status.state.phase === 'error') {
    return <ErrorState message={status.state.message} onRetry={status.retry} />
  }
  const { start, end } = status.state.data.seasons
  const selected = season ?? end

  return (
    <>
      <section className="card">
        <label className="field">
          <span className="field-label">Season</span>
          <select
            className="select"
            value={selected}
            onChange={(event) => setSeason(Number(event.target.value))}
          >
            {Array.from({ length: end - start + 1 }, (_, i) => end - i).map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
      </section>

      {state.phase === 'idle' || state.phase === 'loading' ? (
        state.phase === 'loading' && state.total > 0 ? (
          <ProgressState label={state.label} done={state.done} total={state.total} />
        ) : (
          <Skeleton rows={6} />
        )
      ) : null}
      {state.phase === 'error' ? (
        <ErrorState message={state.message} onRetry={retry} />
      ) : null}
      {state.phase === 'ready' ? (
        <RaceHistoryTable races={state.races} />
      ) : null}
    </>
  )
}

function RaceHistoryTable({ races }: { races: RaceResult[] }) {
  const raced = races.filter((race) => race.hasActuals)
  const hits = raced.filter((race) => race.winnerHit).length
  const avgOverlap =
    raced.length > 0
      ? raced.reduce((sum, race) => sum + (race.top3Overlap ?? 0), 0) / raced.length
      : null

  return (
    <>
      <section className="card">
        <div className="summary-line">
          <span>
            {raced.length} of {races.length} races with actuals
          </span>
          <span>·</span>
          <span>
            Winner hits: {hits}/{raced.length}
          </span>
          {avgOverlap !== null ? (
            <>
              <span>·</span>
              <span>Avg top-3 overlap: {avgOverlap.toFixed(2)}</span>
            </>
          ) : null}
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
                  <td>{race.raceName}</td>
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
