import { useState } from 'react'
import { getStatus, type Job } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { EmptyState } from '../ui/DataState'
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
        buildPayload={() => ({ ...seasonPayload(range), refresh })}
        options={
          <>
            {/* Fetch is the one place that pulls NEW seasons, so the end
                ceiling is the configured end season (not the cached max) and
                the start floor is the configured start (not the modern-era
                clamp), so older seasons like 2010 can be fetched too. */}
            <SeasonRange
              value={range}
              onChange={setRange}
              min={seasons?.start}
              max={seasons?.end}
            />
            <RefreshToggle value={refresh} onChange={setRefresh} />
          </>
        }
        renderResult={(job) => <FetchResult job={job} />}
      />
      <p className="muted config-intro">
        Downloads raw race data from the Jolpica API into <code>data/raw</code>{' '}
        (cached — the pipeline runs offline afterwards). Pick the seasons to
        fetch; every later step (train, backtest) reads
        from this cache, so fetch new seasons here first. Leave the range blank
        to fetch the configured <code>[data]</code> season range. Runs as a
        background job; only one pipeline step runs at a time.
      </p>
      {status.state.phase === 'ready' && !status.state.data.data.has_raw_cache ? (
        <EmptyState title="No raw data cached yet">
          Run <strong>Fetch data</strong> above to download race results from
          the Jolpica API — every later pipeline step (train, backtest, predict)
          reads from that cache and works offline afterwards.
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
