import { METRICS } from './lib'
import { MetricCell } from './MetricCell'

/** One compared model's snapshot inside a backtest job result. */
interface ModelSnapshot {
  overall: Record<string, Record<string, number>>
}

function stemOf(path: string): string {
  return path.split(/[\\/]/).pop()?.replace(/\.joblib$/, '') ?? path
}

/**
 * The result block of a finished backtest job: one row per compared model
 * (when several were selected), plus the first model's numbers vs the
 * baselines. The compared-models table is color-coded so the best/worst
 * model per metric is visible at a glance.
 */
export function BacktestRunResult({ job }: { job: { result: Record<string, unknown> | null; log: string[] } }) {
  const overall = (job.result?.overall ?? {}) as Record<string, Record<string, number>>
  const checkpoint = job.result?.checkpoint as string | undefined
  // The job result carries the per-model data under snapshot.models
  // ({stem: {overall, by_season}}); the top-level `models` key is only the
  // sorted name list.
  const snapshot = (job.result?.snapshot ?? {}) as {
    models?: Record<string, ModelSnapshot>
  }
  const models = snapshot.models ?? {}
  const modelNames = Object.keys(models).sort()
  const names = Object.keys(overall)
  const firstModel = checkpoint ? stemOf(checkpoint) : null

  return (
    <div className="result-block">
      <h3 className="card-title">Backtest run</h3>
      {job.log.length > 0 && (
        <details className="job-log">
          <summary>Log</summary>
          {job.log.map((line, i) => (
            <pre key={i} className="log-line">{line}</pre>
          ))}
        </details>
      )}

      {modelNames.length > 1 ? (
        <>
          <h4 className="card-title">Compared models (mean)</h4>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  {METRICS.map((metric) => (
                    <th key={metric.key} scope="col" className="num">
                      {metric.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {modelNames.map((name) => (
                  <tr key={name}>
                    <td>
                      <code className="mono">{name}</code>
                    </td>
                    {METRICS.map((metric) => (
                      <MetricCell
                        key={metric.key}
                        metric={metric.key}
                        values={modelNames.map(
                          (n) => models[n]?.overall.model?.[metric.key] ?? NaN,
                        )}
                        value={models[name]?.overall.model?.[metric.key]}
                      />
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="cell-tone-legend">
            Green = best model for that metric, red = worst, orange = in
            between, white = tied. MAE is lower-better; the other metrics are
            higher-better.
          </p>
        </>
      ) : null}

      <h4 className="card-title">
        {firstModel ? `${firstModel} vs baselines (mean)` : 'Baselines (mean)'}
      </h4>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Baseline</th>
              {METRICS.map((metric) => (
                <th key={metric.key} scope="col" className="num">
                  {metric.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {names.map((name) => (
              <tr key={name}>
                <td>{name}</td>
                {METRICS.map((metric) => (
                  <MetricCell
                    key={metric.key}
                    metric={metric.key}
                    values={names.map((n) => overall[n]?.[metric.key] ?? NaN)}
                    value={overall[name]?.[metric.key]}
                  />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="cell-tone-legend">
        Green = best in the column, red = worst, orange = in between, white =
        tied. MAE is lower-better; the other metrics are higher-better.
      </p>
    </div>
  )
}
