import { useState } from 'react'
import {
  getCalibration,
  getModels,
  getStatus,
  type Calibration,
  type CalibrationTarget,
  type ModelsResponse,
} from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { fmtNumber } from '../../lib/format'
import { Badge } from '../ui/Badge'
import { Chart, type ChartDatum, type ChartSeries } from '../ui/Chart'
import { ErrorState, Skeleton } from '../ui/DataState'
import { JobRunner } from '../ui/JobRunner'
import { PrereqHint } from '../ui/PrereqHint'
import {
  DEFAULT_SEASON_RANGE,
  SeasonRange,
  seasonPayload,
  type SeasonRangeValue,
} from '../ui/SeasonRange'
import { deployedName, modelChoices } from './lib'

const TARGET_LABEL: Record<string, string> = {
  scored: 'P scored',
  top3: 'P top 3',
  win: 'P win',
}

const DEFAULT_CALIBRATION_MODEL = ''

/**
 * Calibration (CLI `f1 calibrate`) folded into the Backtest tab: fit
 * isotonic probability calibrators for a model and review the deployment
 * decision — per-target raw vs calibrated Brier plus reliability curves.
 * Picking no model fits the shared walk-forward calibrators; picking a saved
 * model calibrates its checkpoint (its own features) and writes calibrators
 * next to it. fit-through/eval-from overrides the hold-out split evaluation.
 */
export function Calibration() {
  const [modelChoice, setModelChoice] = useState<string>(DEFAULT_CALIBRATION_MODEL)
  const [range, setRange] = useState<SeasonRangeValue>(DEFAULT_SEASON_RANGE)
  const [fitThrough, setFitThrough] = useState('')
  const [evalFrom, setEvalFrom] = useState('')
  const [version, setVersion] = useState(0)
  const status = useApi('status', () => getStatus())
  const modelsState = useApi<ModelsResponse>('models', getModels)
  const models = modelsState.state.phase === 'ready' ? modelsState.state.data : null
  const { state, retry } = useApi(`calibration-${version}`, () => getCalibration())

  return (
    <>
      <section className="card">
        <JobRunner
          type="calibrate"
          runLabel="Run calibration"
          onDone={() => setVersion((v) => v + 1)}
          buildPayload={() => ({
            ...seasonPayload(range),
            ...(modelChoice !== DEFAULT_CALIBRATION_MODEL
              ? { model_path: models?.models[modelChoice]?.checkpoint ?? undefined }
              : {}),
            ...(fitThrough.trim() !== '' ? { fit_through_season: Number(fitThrough) } : {}),
            ...(evalFrom.trim() !== '' ? { eval_from_season: Number(evalFrom) } : {}),
          })}
          options={
            <>
              <div className="job-option">
                <label className="job-label" htmlFor="calibrate-model">
                  Model
                </label>
                <select
                  id="calibrate-model"
                  className="select"
                  value={modelChoice}
                  onChange={(e) => setModelChoice(e.target.value)}
                >
                  <option value={DEFAULT_CALIBRATION_MODEL}>
                    config default (deployed)
                  </option>
                  {modelChoices(models)
                    .filter((choice) => choice.value !== 'default')
                    .map((choice) => (
                      <option key={choice.value} value={choice.value}>
                        {choice.label}
                      </option>
                    ))}
                </select>
                <p className="job-option-hint">
                  No model = shared walk-forward calibrators ({' '}
                  {deployedName(models) ?? 'config'} checkpoint's features).
                  A saved model is calibrated on its own out-of-sample seasons
                  (calibrators written next to it) with its own features.
                </p>
              </div>
              <SeasonRange value={range} onChange={setRange} />
              <details className="advanced-options">
                <summary>Advanced</summary>
                <div className="job-options-inner">
                  <div className="season-config">
                    <div className="field">
                      <label className="field-label" htmlFor="cal-fit-through">
                        Fit through season
                      </label>
                      <input
                        id="cal-fit-through"
                        type="number"
                        className="select"
                        placeholder="auto (2/3 split)"
                        value={fitThrough}
                        onChange={(e) => setFitThrough(e.target.value)}
                      />
                    </div>
                    <div className="field">
                      <label className="field-label" htmlFor="cal-eval-from">
                        Evaluate from season
                      </label>
                      <input
                        id="cal-eval-from"
                        type="number"
                        className="select"
                        placeholder="auto (2/3 split)"
                        value={evalFrom}
                        onChange={(e) => setEvalFrom(e.target.value)}
                      />
                    </div>
                  </div>
                  <p className="job-option-hint">
                    Both together override the hold-out split used for the
                    deployment decision (fit OOS seasons ≤ fit-through,
                    evaluate seasons ≥ eval-from). Left blank, the default
                    chronological two-thirds split applies.
                  </p>
                </div>
              </details>
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
          Calibrators that do not improve the hold-out Brier are not deployed.
          If the model was trained through the dataset end, retrain it with an
          earlier end season first — calibration needs out-of-sample seasons.
        </p>
      </section>
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
        <li>
          deployed: <code className="mono">{String(result.deployed)}</code>
        </li>
        <li>
          calibrators: <code className="mono">{String(result.calibrators)}</code>
        </li>
      </ul>
    </div>
  )
}

/** The reports/calibration.json snapshot: Brier table + reliability curves. */
export function CalibrationView({ calibration }: { calibration: Calibration }) {
  const entries = Object.entries(calibration.targets)
  return (
    <>
      <section className="card">
        <h2 className="card-title">Brier score — raw vs calibrated</h2>
        <p className="context-note">
          {calibration.context || 'Calibration evaluation (out-of-sample).'}
        </p>
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
