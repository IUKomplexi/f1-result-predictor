import { useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { getCalibration, type Calibration, type CalibrationTarget } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { fmtNumber } from '../../lib/format'
import { Badge } from '../ui/Badge'
import { ErrorState, Skeleton } from '../ui/DataState'
import { JobRunner } from '../ui/JobRunner'
import './Calibration.css'

const TARGET_LABEL: Record<string, string> = {
  scored: 'P scored',
  top3: 'P top 3',
  win: 'P win',
}

export function Calibration() {
  const [fitThrough, setFitThrough] = useState('')
  const [evalFrom, setEvalFrom] = useState('')
  const [version, setVersion] = useState(0)
  const { state, retry } = useApi(`calibration-${version}`, () => getCalibration())

  return (
    <>
      <JobRunner
        type="calibrate"
        runLabel="Run calibration"
        onDone={() => setVersion((v) => v + 1)}
        buildPayload={() => ({
          fit_through_season: fitThrough === '' ? null : Number(fitThrough),
          eval_from_season: evalFrom === '' ? null : Number(evalFrom),
        })}
        options={
          <div className="season-config">
            <div className="job-option">
              <label className="field-label" htmlFor="cal-fit-through">Fit through season</label>
              <input
                id="cal-fit-through"
                type="number"
                value={fitThrough}
                onChange={(e) => setFitThrough(e.target.value)}
                placeholder="auto"
              />
            </div>
            <div className="job-option">
              <label className="field-label" htmlFor="cal-eval-from">Evaluate from season</label>
              <input
                id="cal-eval-from"
                type="number"
                value={evalFrom}
                onChange={(e) => setEvalFrom(e.target.value)}
                placeholder="auto"
              />
            </div>
          </div>
        }
        renderResult={(job) => <CalibrateRunResult job={job} />}
      />
      <p className="muted config-intro">
        The hold-out split (fit through / evaluate from) controls which
        out-of-sample seasons calibrators are fit on and which they are
        evaluated on for the deployment decision. Leave blank for the default
        chronological two-thirds split. This configures the run — it does not
        change the model.
      </p>
      {state.phase === 'loading' ? (
        <Skeleton rows={6} />
      ) : state.phase === 'error' ? (
        <ErrorState message={state.message} onRetry={retry} />
      ) : (
        <CalibrationView calibration={state.data} />
      )}
    </>
  )
}

function CalibrateRunResult({ job }: { job: { result: Record<string, unknown> | null; log: string[] } }) {
  const result = job.result ?? {}
  return (
    <div className="result-block">
      <h3 className="card-title">Calibration run</h3>
      {job.log.length > 0 && (
        <details className="job-log">
          <summary>Log</summary>
          {job.log.map((line, i) => (
            <pre key={i} className="log-line">{line}</pre>
          ))}
        </details>
      )}
      <ul className="summary-list">
        <li>deployed: <code className="mono">{String(result.deployed)}</code></li>
        <li>fingerprint: <code className="mono">{String(result.fingerprint)}</code></li>
      </ul>
    </div>
  )
}

function CalibrationView({ calibration }: { calibration: Calibration }) {
  const entries = Object.entries(calibration.targets)
  return (
    <>
      <section className="card">
        <h2 className="card-title">Brier score — raw vs calibrated</h2>
        <p className="context-note">{calibration.context || 'Calibration evaluation (out-of-sample).'}</p>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Target</th>
                <th scope="col" className="num">Deployed</th>
                <th scope="col" className="num">Raw</th>
                <th scope="col" className="num">Calibrated</th>
                <th scope="col" className="num">Δ</th>
              </tr>
            </thead>
            <tbody>
              {entries.map(([target, value]) => (
                <tr key={target}>
                  <td>{TARGET_LABEL[target] ?? target}</td>
                  <td className="num">
                    {value.deployed ? (
                      <Badge variant="ready">Yes</Badge>
                    ) : (
                      <Badge variant="missing">No</Badge>
                    )}
                  </td>
                  <td className="num">{fmtNumber(value.brier_raw, 4)}</td>
                  <td className="num">{fmtNumber(value.brier_calibrated, 4)}</td>
                  <td className="num">
                    <span className={value.delta <= 0 ? 'delta-good' : 'delta-bad'}>
                      {fmtNumber(value.delta, 4)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <h2 className="card-title">Reliability curves</h2>
        <p className="context-note">
          Predicted probability (x) vs observed frequency (y). The dashed line is
          perfect calibration.
        </p>
        <div className="chart-grid">
          {entries.map(([target, value]) => (
            <figure key={target} className="chart-figure">
              <figcaption>{TARGET_LABEL[target] ?? target}</figcaption>
              <ReliabilityChart target={value} />
            </figure>
          ))}
        </div>
      </section>
    </>
  )
}

function ReliabilityChart({ target }: { target: CalibrationTarget }) {
  const data = target.reliability.map((bin) => ({
    predicted: bin.mean_pred,
    observed: bin.observed,
  }))
  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
          <CartesianGrid stroke="#2e2e3a" strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="predicted"
            domain={[0, 1]}
            tick={{ fill: '#a8a8b5', fontSize: 11 }}
            stroke="#2e2e3a"
          />
          <YAxis
            type="number"
            domain={[0, 1]}
            tick={{ fill: '#a8a8b5', fontSize: 11 }}
            stroke="#2e2e3a"
          />
          <Tooltip
            contentStyle={{
              background: '#23232e',
              border: '1px solid #2e2e3a',
              borderRadius: 8,
              color: '#f2f2f5',
            }}
          />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke="#6f6f7d"
            strokeDasharray="4 4"
          />
          <Line
            type="linear"
            dataKey="observed"
            name="Observed"
            stroke="#e10600"
            strokeWidth={2}
            dot={{ r: 3, fill: '#e10600' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
