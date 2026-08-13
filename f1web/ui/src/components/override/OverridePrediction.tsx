import { useEffect, useState } from 'react'
import {
  getModels,
  postPrediction,
  type ModelsResponse,
  type Prediction,
  type PredictOverrides,
} from '../../api/client'
import { FeatureToggles, NO_FEATURE_OVERRIDES, type FeatureOverride } from '../ui/FeatureToggles'
import { driverLabel, fmtDate, fmtPoints } from '../../lib/format'
import { Badge } from '../ui/Badge'
import './OverridePrediction.css'

/**
 * Run a one-off prediction with ephemeral overrides (season/round, grid CSV,
 * feature toggles, refresh, model checkpoint, optional report file). Nothing
 * is written to config.toml; overrides apply to this request only.
 */
export function OverridePrediction() {
  const [season, setSeason] = useState<string>('')
  const [round, setRound] = useState<string>('')
  const [grid, setGrid] = useState<string>('')
  const [features, setFeatures] = useState<FeatureOverride>(NO_FEATURE_OVERRIDES)
  const [refresh, setRefresh] = useState(false)
  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [modelChoice, setModelChoice] = useState('default')
  const [customPath, setCustomPath] = useState('')
  const [writeReport, setWriteReport] = useState(false)
  const [pred, setPred] = useState<Prediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getModels().then(setModels).catch(() => {})
  }, [])

  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      const overrides: PredictOverrides = {}
      if (season !== '') overrides.season = Number(season)
      if (round !== '') overrides.round = Number(round)
      if (grid.trim() !== '') overrides.grid_csv = grid
      if (features.enable.length > 0) overrides.enable_features = features.enable
      if (features.disable.length > 0) overrides.disable_features = features.disable
      if (refresh) overrides.refresh = true
      const checkpoint =
        modelChoice === 'custom'
          ? customPath.trim()
          : modelChoice === 'default'
            ? null
            : models?.models[modelChoice]?.checkpoint
      if (checkpoint) overrides.model_path = checkpoint
      if (writeReport) overrides.write_report = true
      const result = await postPrediction(overrides)
      setPred(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  const modelNames = models ? Object.keys(models.models).sort() : []

  return (
    <section className="card">
      <h2 className="card-title">Specific race prediction</h2>
      <p className="muted config-intro">
        Run a one-off prediction with ephemeral overrides — season/round, a
        qualifying grid (CSV text with <code>driver_id,grid</code>), feature
        toggles, a model checkpoint, and optionally the same Markdown report{' '}
        <code>f1 predict</code> writes. Nothing is written to{' '}
        <code>config.toml</code>.
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
      <div className="override-run-options">
        <div className="field">
          <label className="field-label" htmlFor="ov-model">Model</label>
          <select
            id="ov-model"
            className="select"
            value={modelChoice}
            onChange={(e) => setModelChoice(e.target.value)}
          >
            <option value="default">config default</option>
            {modelNames.map((name) => (
              <option key={name} value={name}>{name}</option>
            ))}
            <option value="custom">custom path…</option>
          </select>
        </div>
        {modelChoice === 'custom' ? (
          <div className="field">
            <label className="field-label" htmlFor="ov-model-path">Checkpoint path</label>
            <input
              id="ov-model-path"
              type="text"
              value={customPath}
              onChange={(e) => setCustomPath(e.target.value)}
              placeholder="data/model/other.joblib"
            />
          </div>
        ) : null}
        <div className="field override-checks">
          <label className="check-line">
            <input
              type="checkbox"
              checked={refresh}
              onChange={(e) => setRefresh(e.target.checked)}
            />
            Refresh raw data (ignore cache)
          </label>
          <label className="check-line">
            <input
              type="checkbox"
              checked={writeReport}
              onChange={(e) => setWriteReport(e.target.checked)}
            />
            Write reports/prediction.md
          </label>
        </div>
      </div>
      <FeatureToggles value={features} onChange={setFeatures} />
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
