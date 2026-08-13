import { useState } from 'react'
import { getCalibration, getStatus, type Calibration, type CalibrationTarget } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { fmtNumber } from '../../lib/format'
import { Badge } from '../ui/Badge'
import { Chart, type ChartDatum, type ChartSeries } from '../ui/Chart'
import { ErrorState, Skeleton } from '../ui/DataState'
import { JobRunner } from '../ui/JobRunner'
import { ModelPicker, modelCheckpointPath, useModels, type ModelSelection } from '../ui/ModelPicker'
import { PrereqHint } from '../ui/PrereqHint'
import {
  DEFAULT_SEASON_RANGE,
  SeasonRange,
  seasonPayload,
  type SeasonRangeValue,
} from '../ui/SeasonRange'
import './Calibration.css'

const TARGET_LABEL: Record<string, string> = {
  scored: 'P scored',
  top3: 'P top 3',
  win: 'P win',
}

/**
 * Calibration: pick a trained model and fit isotonic calibrators to its
 * out-of-sample scores. The model's own features are used; it is always
 * evaluated on the newest season (fit on all earlier out-of-sample seasons,
 * deployment decision on the newest one). Nothing else needs configuring.
 */
export function Calibration() {
  const [modelChoice, setModelChoice] = useState<ModelSelection>('default')
  const [range, setRange] = useState<SeasonRangeValue>(DEFAULT_SEASON_RANGE)
  const [version, setVersion] = useState(0)
  const status = useApi('status', () => getStatus())
  const { state: modelsState } = useModels()
  const models = modelsState.phase === 'ready' ? modelsState.data : null
  const { state, retry } = useApi(`calibration-${version}`, () => getCalibration())

  return (
    <>
      <JobRunner
        type="calibrate"
        runLabel="Run calibration"
        onDone={() => setVersion((v) => v + 1)}
        buildPayload={() => ({
          ...seasonPayload(range),
          model_path: modelCheckpointPath(models, modelChoice),
        })}
        options={
          <>
            <ModelPicker
              value={modelChoice}
              onChange={setModelChoice}
              hint="Calibrators are fit on this model's out-of-sample scores and written next to it (data/model/<name>.calibrators.joblib)."
            />
            <SeasonRange value={range} onChange={setRange} />
          </>
        }
        renderResult={(job) => <CalibrateRunResult job={job} />}
      />
      <PrereqHint
        when={status.state.phase === 'ready' && !status.state.data.model.has_checkpoint}
      >
        No model checkpoint yet — run Train first so calibration has a model to
        evaluate.
      </PrereqHint>
      <p className="muted config-intro">
        A model is always judged on the newest season: calibrators are fit on
        every out-of-sample season before it (after the model's training
        window) and the deployment decision — calibrate or keep raw — is made
        on that newest season only. Calibrators that do not improve the
        hold-out Brier are not deployed. If the model was trained through the
        dataset end, retrain it with an earlier end season first.
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
        <li>calibrators: <code className="mono">{String(result.calibrators)}</code></li>
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
  const data: ChartDatum[] = target.reliability.map((bin) => ({
    predicted: bin.mean_pred,
    observed: bin.observed,
  }))
  const series: ChartSeries[] = [
    { key: 'observed', name: 'Observed', color: '#e10600', strokeWidth: 2, dot: true },
  ]
  return (
    <Chart
      data={data}
      xKey="predicted"
      xType="number"
      xDomain={[0, 1]}
      yDomain={[0, 1]}
      series={series}
      referenceLine={{ from: [0, 0], to: [1, 1] }}
      valueFormat={(v) => fmtNumber(v, 4)}
    />
  )
}
