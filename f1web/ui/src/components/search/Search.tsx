import { useState } from 'react'
import { getConfig, putConfig, type Job } from '../../api/client'
import { JobRunner } from '../ui/JobRunner'

export function Search() {
  const [n, setN] = useState(16)
  const [maxTest, setMaxTest] = useState(2019)

  return (
    <JobRunner
      type="search"
      runLabel="Run search"
      buildPayload={() => ({ n, max_test_season: maxTest, seed: 0 })}
      options={
        <>
          <div className="job-option">
            <label className="field-label" htmlFor="search-n">Search n</label>
            <input
              id="search-n"
              type="number"
              min={1}
              value={n}
              onChange={(e) => setN(Number(e.target.value) || 1)}
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
        </>
      }
      renderResult={(job) => <SearchResult job={job} />}
    />
  )
}

function SearchResult({ job }: { job: Job }) {
  const result = job.result ?? {}
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
    <div className="result-block">
      <h3 className="card-title">Search results</h3>
      {job.log.length > 0 && (
        <details className="job-log">
          <summary>Log</summary>
          {job.log.map((line, i) => (
            <pre key={i} className="log-line">{line}</pre>
          ))}
        </details>
      )}
      <p className="summary-list">
        Best config:{' '}
        <code className="mono">
          {Object.entries(best).map(([k, v]) => `${k}=${v}`).join(', ')}
        </code>{' '}
        <button type="button" className="link-button" onClick={apply}>
          Apply to [model.params]
        </button>
      </p>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              {rows[0]
                ? Object.keys(rows[0]).map((k) => (
                    <th key={k} scope="col" className="num">{k}</th>
                  ))
                : null}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {Object.values(row).map((v, j) => (
                  <td key={j} className="num">
                    {typeof v === 'number' ? v.toFixed(3) : String(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
