import { useEffect, useState } from 'react'
import { getCalendar, getPrediction, getStatus, type Prediction } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { driverLabel, fmtDate, fmtPoints } from '../../lib/format'
import { Badge } from '../ui/Badge'
import { ErrorState, Skeleton } from '../ui/DataState'
import { ProbabilityBar } from '../ui/ProbabilityBar'
import './Race.css'

/**
 * Race view: one race's prediction at a time, with a season selector and
 * prev/next round navigation. Defaults to the upcoming "next race"; prior
 * completed rounds show their verified prediction.
 */
export function Race() {
  const status = useApi('status', () => getStatus())
  const [season, setSeason] = useState<number | null>(null)
  const [round, setRound] = useState<number | null>(null)
  const [rounds, setRounds] = useState<number[]>([])
  const [primed, setPrimed] = useState(false)

  // Default to the global "next race" once status is ready.
  useEffect(() => {
    if (primed || status.state.phase !== 'ready') return
    setPrimed(true)
    const fallbackSeason = status.state.data.seasons.end
    getPrediction()
      .then((pred) => {
        setSeason(pred.season)
        setRound(pred.round)
      })
      .catch(() => {
        // Fall back to the newest configured season (no round -> snapped below).
        setSeason(fallbackSeason)
        setRound(null)
      })
  }, [primed, status.state.phase])

  // Load the ordered round list whenever the season changes.
  useEffect(() => {
    if (season === null) return
    let cancelled = false
    getCalendar(season)
      .then((data) => {
        if (cancelled) return
        setRounds(data.calendar.map((c) => c.round).sort((a, b) => a - b))
      })
      .catch(() => {
        if (!cancelled) setRounds([])
      })
    return () => {
      cancelled = true
    }
  }, [season])

  // Snap to a valid round: if none is selected or the current one is no longer
  // in this season's calendar, fall back to the most recent round.
  useEffect(() => {
    if (season === null || rounds.length === 0) return
    if (round === null || !rounds.includes(round)) {
      setRound(rounds[rounds.length - 1])
    }
  }, [season, round, rounds])

  if (status.state.phase === 'loading') return <Skeleton rows={8} />
  if (status.state.phase === 'error') {
    return <ErrorState message={status.state.message} onRetry={status.retry} />
  }

  const { start, end } = status.state.data.seasons
  const selected = season ?? end
  const configured = Array.from({ length: end - start + 1 }, (_, i) => end - i)
  // The default next race can be in a brand-new season beyond the configured
  // range; keep its season selectable.
  const seasons = configured.includes(selected) ? configured : [selected, ...configured]
  const idx = round !== null ? rounds.indexOf(round) : -1
  const canPrev = idx > 0
  const canNext = idx >= 0 && idx < rounds.length - 1

  return (
    <>
      <section className="card">
        <div className="race-nav">
          <label className="field">
            <span className="field-label">Season</span>
            <select
              className="select"
              value={selected}
              onChange={(event) => {
                setSeason(Number(event.target.value))
                setRound(null)
              }}
            >
              {seasons.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <div className="pager">
            <button
              type="button"
              className="button"
              disabled={!canPrev}
              onClick={() => setRound(rounds[idx - 1])}
            >
              ‹ Prev
            </button>
            <span className="pager-label">{round !== null ? `Round ${round}` : '—'}</span>
            <button
              type="button"
              className="button"
              disabled={!canNext}
              onClick={() => setRound(rounds[idx + 1])}
            >
              Next ›
            </button>
          </div>
        </div>
      </section>

      {season === null || round === null ? (
        <Skeleton rows={10} />
      ) : (
        <RacePanel season={season} round={round} />
      )}
    </>
  )
}

function RacePanel({ season, round }: { season: number; round: number }) {
  const [refresh, setRefresh] = useState(false)
  const { state, retry } = useApi(
    `race-${season}-${round}-${refresh}`,
    () => getPrediction(season, round, refresh),
  )
  if (state.phase === 'loading') return <Skeleton rows={10} />
  if (state.phase === 'error') return <ErrorState message={state.message} onRetry={retry} />
  return (
    <>
      <section className="card race-refresh-row">
        <label
          className="check-line"
          title="Ignore the raw-data cache and re-fetch from the API (CLI --refresh)."
        >
          <input
            type="checkbox"
            checked={refresh}
            onChange={(e) => setRefresh(e.target.checked)}
          />
          Refresh raw data (ignore cache)
        </label>
      </section>
      <RaceTable prediction={state.data} />
    </>
  )
}

function RaceTable({ prediction }: { prediction: Prediction }) {
  const { race, drivers, synthetic, verified, calibrated } = prediction
  return (
    <>
      <section className="card">
        <div className="race-meta">
          <div>
            <h2 className="card-title">
              {race.race_name ?? `Round ${prediction.round}`}
            </h2>
            <p className="meta-line">
              <span>
                Round {prediction.round} · season {prediction.season}
              </span>
              <span>·</span>
              <span>{fmtDate(race.date)}</span>
              {race.circuit_id ? (
                <>
                  <span>·</span>
                  <span>{driverLabel(race.circuit_id)}</span>
                </>
              ) : null}
            </p>
          </div>
          <div className="badge-row">
            {synthetic ? (
              <Badge variant="warn">Unverified · synthetic grid</Badge>
            ) : null}
            {verified ? <Badge variant="ready">Has actuals</Badge> : null}
            {calibrated ? <Badge variant="info">Calibrated probabilities</Badge> : null}
          </div>
        </div>
      </section>

      <section className="card">
        <h2 className="card-title">Ranked grid</h2>
        <div className="table-wrap">
          <table className="data-table grid-table">
            <thead>
              <tr>
                <th scope="col" className="num">#</th>
                <th scope="col">Driver</th>
                <th scope="col" className="hide-narrow">Team</th>
                <th scope="col" className="num">Grid</th>
                <th scope="col" className="num">Exp. pts</th>
                <th scope="col" className="num">P scored</th>
                <th scope="col" className="num">P top 3</th>
                <th scope="col" className="num">P win</th>
                {verified ? <th scope="col" className="num">Actual</th> : null}
              </tr>
            </thead>
            <tbody>
              {drivers.map((row) => (
                <tr key={row.driver_id}>
                  <td className="num rank">{row.pred_rank}</td>
                  <td className="driver">{driverLabel(row.driver_id)}</td>
                  <td className="muted hide-narrow">{driverLabel(row.constructor_id)}</td>
                  <td className="num">{row.grid ?? '–'}</td>
                  <td className="num">{fmtPoints(row.expected_points)}</td>
                  <td className="num">
                    <ProbabilityBar value={row.p_scored} label="P scored" variant="scored" />
                  </td>
                  <td className="num">
                    <ProbabilityBar value={row.p_top3} label="P top 3" variant="top3" />
                  </td>
                  <td className="num">
                    <ProbabilityBar value={row.p_win} label="P win" variant="win" />
                  </td>
                  {verified ? (
                    <td className="num">
                      {row.actual_position ? `P${row.actual_position}` : '–'} ·{' '}
                      {fmtPoints(row.actual_points)}
                    </td>
                  ) : null}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  )
}
