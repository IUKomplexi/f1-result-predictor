import type { Job } from '../../api/client'
import { JobRunner } from '../ui/JobRunner'

export function Train() {
  return (
    <JobRunner
      type="train"
      runLabel="Train model"
      buildPayload={() => ({})}
      renderResult={(job) => <TrainResult job={job} />}
    />
  )
}

function TrainResult({ job }: { job: Job }) {
  const result = job.result ?? {}
  return (
    <div className="result-block">
      <h3 className="card-title">Training summary</h3>
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
          <strong>{String(result.rows)}</strong> rows ·{' '}
          <strong>{String(result.seasons)}</strong> seasons
        </li>
        <li>
          <strong>{String(result.n_features)}</strong> features (fp{' '}
          <code className="mono">{String(result.fingerprint)}</code>)
        </li>
        <li>checkpoint: <code className="mono">{String(result.checkpoint)}</code></li>
      </ul>
    </div>
  )
}
