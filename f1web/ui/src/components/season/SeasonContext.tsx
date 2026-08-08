import { useState } from 'react'
import { getCalendar, getStandings, getStatus } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { driverLabel, fmtDate } from '../../lib/format'
import { ErrorState, Skeleton } from '../ui/DataState'
import './SeasonContext.css'

export function SeasonContext() {
  const status = useApi('status', () => getStatus())
  const [season, setSeason] = useState<number | null>(null)

  if (status.state.phase === 'loading') return <Skeleton rows={4} />
  if (status.state.phase === 'error') {
    return <ErrorState message={status.state.message} onRetry={status.retry} />
  }
  const end = status.state.data.seasons.end
  const start = status.state.data.seasons.start
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
      <SeasonData season={selected} />
    </>
  )
}

function SeasonData({ season }: { season: number }) {
  const { state, retry } = useApi(`season-${season}`, () =>
    Promise.all([getCalendar(season), getStandings(season)]),
  )
  if (state.phase === 'loading') return <Skeleton rows={6} />
  if (state.phase === 'error') return <ErrorState message={state.message} onRetry={retry} />
  const [calendar, standings] = state.data
  return (
    <>
      <section className="card">
        <h2 className="card-title">Calendar</h2>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col" className="num">Rd</th>
                <th scope="col">Date</th>
                <th scope="col">Race</th>
                <th scope="col" className="hide-narrow">Circuit</th>
                <th scope="col" className="hide-narrow">Country</th>
                <th scope="col" className="num">Sprint</th>
              </tr>
            </thead>
            <tbody>
              {calendar.calendar.map((race) => (
                <tr key={race.round}>
                  <td className="num">{race.round}</td>
                  <td>{fmtDate(race.date)}</td>
                  <td>{race.race_name || `Round ${race.round}`}</td>
                  <td className="muted hide-narrow">
                    {driverLabel(race.circuit_name ?? race.circuit_id)}
                  </td>
                  <td className="muted hide-narrow">{race.country}</td>
                  <td className="num">{race.is_sprint_round ? 'Yes' : '–'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <h2 className="card-title">Driver standings</h2>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col" className="num">Pos</th>
                <th scope="col">Driver</th>
                <th scope="col" className="hide-narrow">Team</th>
                <th scope="col" className="num">Points</th>
                <th scope="col" className="num">Wins</th>
              </tr>
            </thead>
            <tbody>
              {standings.driver.map((row, index) => (
                <tr key={row.driver_id ?? index}>
                  <td className="num">{row.position ?? '–'}</td>
                  <td>{driverLabel(row.driver_id)}</td>
                  <td className="muted hide-narrow">
                    {driverLabel(row.constructor_id)}
                  </td>
                  <td className="num">{row.points}</td>
                  <td className="num">{row.wins ?? '–'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <h2 className="card-title">Constructor standings</h2>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col" className="num">Pos</th>
                <th scope="col">Constructor</th>
                <th scope="col" className="num">Points</th>
                <th scope="col" className="num">Wins</th>
              </tr>
            </thead>
            <tbody>
              {standings.constructor.map((row, index) => (
                <tr key={row.constructor_id ?? index}>
                  <td className="num">{row.position ?? '–'}</td>
                  <td>{driverLabel(row.constructor_id)}</td>
                  <td className="num">{row.points}</td>
                  <td className="num">{row.wins ?? '–'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  )
}
