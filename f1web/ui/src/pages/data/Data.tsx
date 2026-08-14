import { useState } from 'react'
import { getStatus, type Job } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { EmptyState } from '../../components/ui/DataState'
import { JobRunner } from '../../components/jobs/JobRunner'
import {
  DEFAULT_SEASON_RANGE,
  SeasonRange,
  seasonPayload,
  seasonRangeError,
  type SeasonRangeValue,
} from '../../components/controls/SeasonRange'
import { RefreshToggle } from '../../components/controls/RefreshToggle'
import './Data.css'

export function Data() {
  const status = useApi('status', () => getStatus())
  const [range, setRange] = useState<SeasonRangeValue>(DEFAULT_SEASON_RANGE)
  const [refresh, setRefresh] = useState(false)
  const seasons =
    status.state.phase === 'ready' ? status.state.data.seasons : null
  return (
    <>
      <JobRunner
        type="fetch"
        runLabel="Fetch data"
        layout="action-end"
        buildPayload={() => {
          const rangeError = seasonRangeError(range)
          if (rangeError) throw new Error(rangeError)
          return { ...seasonPayload(range), refresh }
        }}
        options={
          <>
            {/* Fetch is the one place that pulls NEW seasons, so the end
                ceiling is the configured end season (not the cached max) and
                the start floor is the configured start (not the modern-era
                clamp), so older seasons down to 2014 can be fetched too. */}
            <SeasonRange
              value={range}
              onChange={setRange}
              min={seasons?.start}
              max={seasons?.end}
            />
            <RefreshToggle value={refresh} onChange={setRefresh} label="Re-fetch from API" />
          </>
        }
        renderResult={(job) => <FetchResult job={job} />}
      />
      {status.state.phase === 'ready' && !status.state.data.data.has_raw_cache ? (
        <EmptyState title="No data downloaded yet">
          Click <strong>Fetch data</strong> above first — nothing else can run
          without it.
        </EmptyState>
      ) : null}
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
