import { useState } from 'react'
import type { Job } from '../../api/client'
import { JobRunner } from '../ui/JobRunner'
import {
  DEFAULT_SEASON_RANGE,
  SeasonRange,
  seasonPayload,
  type SeasonRangeValue,
} from '../ui/SeasonRange'
import { RefreshToggle } from '../ui/RefreshToggle'
import './Data.css'

export function Data() {
  const [range, setRange] = useState<SeasonRangeValue>(DEFAULT_SEASON_RANGE)
  const [refresh, setRefresh] = useState(false)
  return (
    <>
      <JobRunner
        type="fetch"
        runLabel="Fetch data"
        buildPayload={() => ({ ...seasonPayload(range), refresh })}
        options={
          <>
            <SeasonRange value={range} onChange={setRange} />
            <RefreshToggle value={refresh} onChange={setRefresh} />
          </>
        }
        renderResult={(job) => <FetchResult job={job} />}
      />
      <p className="muted config-intro">
        Fetches cached raw results from the Jolpica API (offline afterwards) into{' '}
        <code>data/raw</code>. Runs as a background job; only one pipeline step
        runs at a time.
      </p>
    </>
  )
}

function FetchResult({ job }: { job: Job }) {
  const seasons = job.result?.seasons as
    | Record<string, { rounds: number; results: number; sprints: number }>
    | undefined
  return (
    <div className="result-block">
      <h3 className="card-title">Fetched seasons</h3>
      {job.log.length > 0 && (
        <details className="job-log">
          <summary>Log</summary>
          {job.log.map((line, i) => (
            <pre key={i} className="log-line">{line}</pre>
          ))}
        </details>
      )}
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Season</th>
              <th scope="col" className="num">Rounds</th>
              <th scope="col" className="num">Results</th>
              <th scope="col" className="num">Sprints</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(seasons ?? {}).map(([season, s]) => (
              <tr key={season}>
                <td>{season}</td>
                <td className="num">{s.rounds}</td>
                <td className="num">{s.results}</td>
                <td className="num">{s.sprints}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
