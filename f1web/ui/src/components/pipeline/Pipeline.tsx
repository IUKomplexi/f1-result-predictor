import { useEffect, useState } from 'react'
import {
  getConfig,
  postJob,
  postPrediction,
  putConfig,
  type ConfigResponse,
  type Job,
  type JobSummary,
  type Prediction,
  type PredictionRow,
  type PredictOverrides,
} from '../../api/client'
import { useJob, useJobs } from '../../hooks/useJob'
import { driverLabel, fmtDate, fmtPoints } from '../../lib/format'
import { Badge } from '../ui/Badge'
import './Pipeline.css'

const JOB_OPTIONS = [
  { type: 'fetch', label: 'Fetch data', payload: {} },
  { type: 'train', label: 'Train', payload: {} },
  { type: 'calibrate', label: 'Calibrate', payload: {} },
  { type: 'backtest', label: 'Backtest', payload: {} },
  { type: 'search', label: 'Search', payload: {} },
] as const

export function Pipeline() {
  return (
    <>
      <JobControls />
      <JobList />
      <JobResults />
      <OverridePrediction />
    </>
  )
}

/* ------------------------------------------------------------- job controls */

function JobControls() {
  const { jobs, refresh } = useJobs()
  const [quantize, setQuantize] = useState(true)
  const [searchN, setSearchN] = useState(16)
  const [maxTest, setMaxTest] = useState(2019)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const busyJob = jobs.some((j) => j.status === 'running' || j.status === 'queued')

  const run = async (type: string, payload: Record<string, unknown>) => {
    setBusy(type)
    setError(null)
    try {
      await postJob(type, payload)
      refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(null)
    }
  }

  const job = (type: string): { label: string; payload: Record<string, unknown> } => {
    if (type === 'backtest') return { label: 'Backtest', payload: { quantize } }
    if (type === 'search')
      return {
        label: 'Search',
        payload: { n: searchN, max_test_season: maxTest, seed: 0 },
      }
    const found = JOB_OPTIONS.find((o) => o.type === type)
    return { label: found?.label ?? type, payload: {} }
  }

  return (
    <section className="card">
      <h2 className="card-title">Pipeline</h2>
      <p className="muted config-intro">
        Run pipeline steps as async background jobs. One job runs at a time;
        the rest queue. Results land in the dashboard below.
      </p>
      <div className="job-options">
        <div className="job-option">
          <label className="check-line">
            <input
              type="checkbox"
              checked={quantize}
              onChange={(e) => setQuantize(e.target.checked)}
            />
            Quantize points
          </label>
        </div>
        <div className="job-option">
          <label className="field-label" htmlFor="search-n">Search n</label>
          <input
            id="search-n"
            type="number"
            min={1}
            value={searchN}
            onChange={(e) => setSearchN(Number(e.target.value) || 1)}
          />
        </div>
        <div className="job-option">
          <label className="field-label" htmlFor="max-test">Max test season</label>
          <input
            id="max-test"
            type="number"
            value={maxTest}
            onChange={(e) => setMaxTest(Number(e.target.value) || 2019)}
          />
        </div>
      </div>
      <div className="job-buttons">
        {JOB_OPTIONS.map((option) => {
          const disabled = busy !== null || busyJob
          return (
            <button
              key={option.type}
              type="button"
              className="button"
              disabled={disabled}
              onClick={() => run(option.type, job(option.type).payload)}
            >
              {busy === option.type ? 'Queued…' : option.label}
            </button>
          )
        })}
      </div>
      {error ? <p className="save-status error">{error}</p> : null}
    </section>
  )
}

/* ------------------------------------------------------------------ job list */

function JobList() {
  const { jobs } = useJobs()
  if (jobs.length === 0) {
    return (
      <section className="card">
        <h2 className="card-title">Jobs</h2>
        <p className="muted">No jobs run yet.</p>
      </section>
    )
  }
  return (
    <section className="card">
      <h2 className="card-title">Job history</h2>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Job</th>
              <th scope="col">Type</th>
              <th scope="col">Status</th>
              <th scope="col" className="num">Started</th>
              <th scope="col" className="num">Finished</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.id}>
                <td className="mono">{job.id}</td>
                <td>{job.label}</td>
                <td>
                  <StatusBadge status={job.status} />
                </td>
                <td className="num muted">{ts(job.started_at)}</td>
                <td className="num muted">{ts(job.finished_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function StatusBadge({ status }: { status: Job['status'] }) {
  if (status === 'done') return <Badge variant="ready">Done</Badge>
  if (status === 'failed') return <Badge variant="missing">Failed</Badge>
  if (status === 'running') return <Badge variant="info">Running</Badge>
  return <Badge variant="warn">Queued</Badge>
}

/* ------------------------------------------------------------------ results */

function JobResults() {
  const { jobs } = useJobs()
  // Newest finished job per type.
  const latest = new Map<string, JobSummary>()
  for (const j of jobs) {
    if (j.status === 'done' && !latest.has(j.type)) latest.set(j.type, j)
  }
  const running = jobs.find((j) => j.status === 'running')
  const runningDetail = useJob(running?.id ?? null)

  return (
    <section className="card">
      <h2 className="card-title">Results</h2>
      {runningDetail && runningDetail.status === 'running' ? (
        <div className="job-log">
          <h3 className="card-title">{runningDetail.label} — running</h3>
          {runningDetail.log.map((line, i) => (
            <pre key={i} className="log-line">{line}</pre>
          ))}
        </div>
      ) : null}
      {latest.size === 0 ? <p className="muted">No finished jobs yet.</p> : null}
      {[...latest.values()].map((job) => (
        <ResultBlock key={job.id} id={job.id} label={job.label} />
      ))}
    </section>
  )
}

function ResultBlock({ id, label }: { id: string; label: string }) {
  const job = useJob(id)
  if (!job || job.status !== 'done') return null
  const result = job.result as Record<string, unknown> | null
  return (
    <div className="result-block">
      <h3 className="card-title">{label}</h3>
      {job.log.length > 0 && (
        <details className="job-log">
          <summary>Log</summary>
          {job.log.map((line, i) => (
            <pre key={i} className="log-line">{line}</pre>
          ))}
        </details>
      )}
      {result ? <ResultBody type={job.type} result={result} /> : <p className="muted">No result payload.</p>}
    </div>
  )
}

function ResultBody({ type, result }: { type: string; result: Record<string, unknown> }) {
  if (type === 'backtest') return <BacktestResult result={result} />
  if (type === 'search') return <SearchResult result={result} />
  if (type === 'calibrate') return <CalibrateResult result={result} />
  if (type === 'fetch') return <FetchResult result={result} />
  return <TrainResult result={result} />
}

function TrainResult({ result }: { result: Record<string, unknown> }) {
  return (
    <ul className="summary-list">
      <li><strong>{String(result.rows)}</strong> rows · <strong>{String(result.seasons)}</strong> seasons</li>
      <li><strong>{String(result.n_features)}</strong> features (fp <code className="mono">{String(result.fingerprint)}</code>)</li>
      <li>checkpoint: <code className="mono">{String(result.checkpoint)}</code></li>
    </ul>
  )
}

function FetchResult({ result }: { result: Record<string, unknown> }) {
  const seasons = (result.seasons ?? {}) as Record<string, { rounds: number; results: number; sprints: number }>
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr><th scope="col">Season</th><th scope="col" className="num">Rounds</th>
            <th scope="col" className="num">Results</th><th scope="col" className="num">Sprints</th></tr>
        </thead>
        <tbody>
          {Object.entries(seasons).map(([season, s]) => (
            <tr key={season}>
              <td>{season}</td><td className="num">{s.rounds}</td>
              <td className="num">{s.results}</td><td className="num">{s.sprints}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function BacktestResult({ result }: { result: Record<string, unknown> }) {
  const overall = (result.overall ?? {}) as Record<string, Record<string, number>>
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr><th scope="col">Baseline</th><th scope="col" className="num">Winner hit</th>
            <th scope="col" className="num">Top3 overlap</th><th scope="col" className="num">Spearman</th>
            <th scope="col" className="num">MAE</th></tr>
        </thead>
        <tbody>
          {Object.entries(overall).map(([name, m]) => (
            <tr key={name}>
              <td>{name}</td>
              <td className="num">{m.winner_hit?.toFixed(3)}</td>
              <td className="num">{m.top3_overlap?.toFixed(3)}</td>
              <td className="num">{m.spearman?.toFixed(3)}</td>
              <td className="num">{m.mae?.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function SearchResult({ result }: { result: Record<string, unknown> }) {
  const best = (result.best ?? {}) as Record<string, number>
  const rows = (result.results ?? []) as Record<string, number>[]
  const apply = async () => {
    const cfg = await getConfig()
    const params = { ...best }
    await putConfig({
      ...cfg.config,
      model: { ...cfg.config.model, params },
    })
  }
  return (
    <>
      <p className="summary-list">
        Best config:{' '}
        <code className="mono">{Object.entries(best).map(([k, v]) => `${k}=${v}`).join(', ')}</code>{' '}
        <button type="button" className="link-button" onClick={apply}>
          Apply to [model.params]
        </button>
      </p>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              {rows[0] ? Object.keys(rows[0]).map((k) => <th key={k} scope="col" className="num">{k}</th>) : null}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {Object.values(row).map((v, j) => (
                  <td key={j} className="num">{typeof v === 'number' ? v.toFixed(3) : String(v)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

function CalibrateResult({ result }: { result: Record<string, unknown> }) {
  const targets = (result.snapshot as { targets?: Record<string, { brier_raw: number; brier_calibrated: number; deployed: boolean }> } | undefined)?.targets ?? {}
  return (
    <>
      <ul className="summary-list">
        <li>deployed: <code className="mono">{String(result.deployed)}</code></li>
        <li>fingerprint: <code className="mono">{String(result.fingerprint)}</code></li>
      </ul>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr><th scope="col">Target</th><th scope="col" className="num">Brier raw</th>
              <th scope="col" className="num">Brier cal</th><th scope="col">Deployed</th></tr>
          </thead>
          <tbody>
            {Object.entries(targets).map(([t, row]) => (
              <tr key={t}>
                <td>{t}</td>
                <td className="num">{row.brier_raw.toFixed(4)}</td>
                <td className="num">{row.brier_calibrated.toFixed(4)}</td>
                <td>{row.deployed ? <Badge variant="ready">yes</Badge> : <Badge variant="warn">no</Badge>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

/* ------------------------------------------------- override prediction */

function OverridePrediction() {
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
      <h2 className="card-title">Prediction with overrides</h2>
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
          <span className="field-label">Features</span>
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
            {drivers.map((row: PredictionRow) => (
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

function ts(epoch: number | null): string {
  if (!epoch) return '–'
  return new Date(epoch * 1000).toLocaleTimeString()
}
