import { METRICS } from './lib'
import { MetricCell } from './MetricCell'

/** The result block of a finished backtest job (overall table + compared models). */
export function BacktestRunResult({ job }: { job: { result: Record<string, unknown> | null; log: string[] } }) {
  const overall = (job.result?.overall ?? {}) as Record<string, Record<string, number>>
  const checkpoint = job.result?.checkpoint as string | undefined
  const compared = (job.result?.models ?? []) as string[]
  const names = Object.keys(overall)
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
      {checkpoint ? (
        <p className="summary-list">
          Model: <code className="mono">{checkpoint}</code>
        </p>
      ) : null}
      {compared.length > 1 ? (
        <p className="summary-list">
          Compared models: <code className="mono">{compared.join(', ')}</code>
        </p>
      ) : null}
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
