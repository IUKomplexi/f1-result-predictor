import { getPrediction, type Prediction } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { driverLabel, fmtDate, fmtPoints } from '../../lib/format'
import { Badge } from '../ui/Badge'
import { ErrorState, Skeleton } from '../ui/DataState'
import { ProbabilityBar } from '../ui/ProbabilityBar'
import './NextRace.css'

export function NextRace() {
  const { state, retry } = useApi('next', () => getPrediction())
  if (state.phase === 'loading') return <Skeleton rows={10} />
  if (state.phase === 'error') return <ErrorState message={state.message} onRetry={retry} />
  return <NextRaceTable prediction={state.data} />
}

function NextRaceTable({ prediction }: { prediction: Prediction }) {
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
