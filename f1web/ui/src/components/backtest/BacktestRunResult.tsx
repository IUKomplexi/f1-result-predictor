/** The result block of a finished backtest job (overall table + compared models). */
export function BacktestRunResult({ job }: { job: { result: Record<string, unknown> | null; log: string[] } }) {
  const overall = (job.result?.overall ?? {}) as Record<string, Record<string, number>>
  const checkpoint = job.result?.checkpoint as string | undefined
  const compared = (job.result?.models ?? []) as string[]
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
              <th scope="col" className="num">Winner hit</th>
              <th scope="col" className="num">Top3 overlap</th>
              <th scope="col" className="num">Spearman</th>
              <th scope="col" className="num">MAE</th>
            </tr>
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
    </div>
  )
}
