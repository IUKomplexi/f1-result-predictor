import type { PredictionRow } from '../../api/client'
import { driverLabel, fmtNumber } from '../../lib/format'

function podiumLabel(ids: string[]): string {
  return ids.length === 0
    ? '–'
    : ids.map((id, i) => `${i + 1}. ${driverLabel(id)}`).join('  ·  ')
}

/**
 * Weekend scoreboard: how the model's prediction for a completed race held up
 * against the actuals. Shown on the Race tab whenever the race is verified.
 * Metrics are computed from the prediction rows themselves (no extra API call).
 */
export function RaceScoreboard({ drivers }: { drivers: PredictionRow[] }) {
  const actuals = drivers.filter(
    (d) => d.actual_position !== null && d.actual_points !== null,
  )
  if (actuals.length === 0) return null

  const predictedTop10 = new Set(
    drivers.filter((d) => d.pred_rank <= 10).map((d) => d.driver_id),
  )
  const actualTop10 = new Set(
    actuals.filter((d) => (d.actual_position as number) <= 10).map((d) => d.driver_id),
  )
  const top10Hits = [...predictedTop10].filter((id) => actualTop10.has(id)).length

  const winnerPick = drivers.find((d) => d.pred_rank === 1)
  const actualWinner = actuals.find((d) => (d.actual_position as number) === 1)
  const winnerHit =
    winnerPick !== undefined &&
    actualWinner !== undefined &&
    winnerPick.driver_id === actualWinner.driver_id

  const predictedTop3 = drivers.filter((d) => d.pred_rank <= 3).map((d) => d.driver_id)
  const actualTop3 = actuals
    .filter((d) => (d.actual_position as number) <= 3)
    .map((d) => d.driver_id)
  const podiumHits = predictedTop3.filter((id) => actualTop3.includes(id)).length

  const mae =
    actuals.reduce(
      (sum, d) => sum + Math.abs((d.actual_points as number) - d.expected_points),
      0,
    ) / actuals.length

  return (
    <section className="card">
      <div className="scoreboard-head">
        <h2 className="card-title">Race scoreboard</h2>
        <span className="muted scoreboard-note">Model vs actuals</span>
      </div>
      <div className="metrics">
        <div className="metric">
          <span className="metric-label">Top-10 hit</span>
          <span className="metric-value">{top10Hits}/10</span>
          <span className="metric-sub">predicted ∩ actual</span>
        </div>
        <div className="metric">
          <span className="metric-label">Podium overlap</span>
          <span className="metric-value">{podiumHits}/3</span>
          <span className="metric-sub">predicted ∩ actual</span>
        </div>
        <div className="metric">
          <span className="metric-label">Winner pick</span>
          <span className={`metric-value ${winnerHit ? 'delta-pos' : 'delta-neg'}`}>
            {winnerHit ? 'Hit' : 'Miss'}
          </span>
          <span className="metric-sub">{winnerPick ? driverLabel(winnerPick.driver_id) : '–'}</span>
        </div>
        <div className="metric">
          <span className="metric-label">MAE</span>
          <span className="metric-value">{fmtNumber(mae, 2)}</span>
          <span className="metric-sub">points · all drivers</span>
        </div>
      </div>
      <div className="podium-line">
        <span className="podium-label">Model podium</span>
        <span>{podiumLabel(predictedTop3)}</span>
      </div>
      <div className="podium-line">
        <span className="podium-label">Actual podium</span>
        <span>{podiumLabel(actualTop3)}</span>
      </div>
    </section>
  )
}
