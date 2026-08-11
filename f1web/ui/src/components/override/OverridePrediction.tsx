import { useEffect, useState } from 'react'
import {
  getConfig,
  postPrediction,
  type ConfigResponse,
  type Prediction,
  type PredictOverrides,
} from '../../api/client'
import { driverLabel, fmtDate, fmtPoints } from '../../lib/format'
import { Badge } from '../ui/Badge'
import './OverridePrediction.css'

/**
 * Run a one-off prediction with ephemeral overrides (season/round, grid CSV,
 * feature toggles). Nothing is written to config.toml; overrides apply to this
 * request only.
 */
export function OverridePrediction() {
  const [season, setSeason] = useState<string>('')
  const [round, setRound] = useState<string>('')
  const [grid, setGrid] = useState<string>('')
  const [pred, setPred] = useState<Prediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [cfg, setCfg] = useState<ConfigResponse | null>(null)

  useEffect(() => {
    getConfig().then(setCfg).catch(() => {})
  }, [])

  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      const overrides: PredictOverrides = {}
      if (season !== '') overrides.season = Number(season)
      if (round !== '') overrides.round = Number(round)
      if (grid.trim() !== '') overrides.grid_csv = grid
      const result = await postPrediction(overrides)
      setPred(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="card">
      <h2 className="card-title">Specific race prediction</h2>
      <p className="muted config-intro">
        Run a one-off prediction with ephemeral overrides — season/round, a
        qualifying grid (CSV text with <code>driver_id,grid</code>), and feature
        toggles. Nothing is written to <code>config.toml</code>.
      </p>
      <div className="override-grid">
        <div className="field">
          <label className="field-label" htmlFor="ov-season">Season</label>
          <input id="ov-season" type="number" value={season} onChange={(e) => setSeason(e.target.value)} placeholder="next race" />
        </div>
        <div className="field">
          <label className="field-label" htmlFor="ov-round">Round</label>
          <input id="ov-round" type="number" value={round} onChange={(e) => setRound(e.target.value)} placeholder="auto" />
        </div>
        <div className="field span-all">
          <label className="field-label" htmlFor="ov-grid">Grid CSV</label>
          <textarea
            id="ov-grid"
            rows={4}
            value={grid}
            onChange={(e) => setGrid(e.target.value)}
            placeholder={'driver_id,grid\nrussell,1\nleclerc,2'}
          />
        </div>
      </div>
      {cfg ? (
        <div className="override-toggles">
          <span className="field-label">Default features</span>
          {cfg.features.registry
            .filter((id) => cfg.features.defaults.includes(id))
            .map((id) => <span key={id} className="mono feature-tag">{id}</span>)}
        </div>
      ) : null}
      <div className="save-row">
        <button type="button" className="button" onClick={run} disabled={loading}>
          {loading ? 'Predicting…' : 'Predict'}
        </button>
      </div>
      {error ? <p className="save-status error">{error}</p> : null}
      {pred ? <PredictionPanel prediction={pred} /> : null}
    </section>
  )
}

function PredictionPanel({ prediction }: { prediction: Prediction }) {
  const { race, drivers, synthetic, verified, calibrated } = prediction
  return (
    <>
      <div className="badge-row">
        {synthetic ? <Badge variant="warn">Unverified · synthetic</Badge> : null}
        {verified ? <Badge variant="ready">Has actuals</Badge> : null}
        {calibrated ? <Badge variant="info">Calibrated</Badge> : null}
      </div>
      <p className="meta-line">
        {race.race_name ?? `Round ${prediction.round}`} · season {prediction.season} ·{' '}
        {fmtDate(race.date)} {race.circuit_id ? `· ${driverLabel(race.circuit_id)}` : ''}
      </p>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col" className="num">#</th>
              <th scope="col">Driver</th>
              <th scope="col">Team</th>
              <th scope="col" className="num">Grid</th>
              <th scope="col" className="num">Exp. pts</th>
            </tr>
          </thead>
          <tbody>
            {drivers.map((row) => (
              <tr key={row.driver_id}>
                <td className="num">{row.pred_rank}</td>
                <td>{driverLabel(row.driver_id)}</td>
                <td className="muted">{driverLabel(row.constructor_id)}</td>
                <td className="num">{row.grid ?? '–'}</td>
                <td className="num">{fmtPoints(row.expected_points)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}
