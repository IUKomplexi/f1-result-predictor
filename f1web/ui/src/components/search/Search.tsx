import { useState } from 'react'
import { getConfig, putConfig, type Job } from '../../api/client'
import { FeatureToggles, NO_FEATURE_OVERRIDES, type FeatureOverride } from '../ui/FeatureToggles'
import { JobRunner } from '../ui/JobRunner'
import { RefreshToggle } from '../ui/RefreshToggle'
import {
  DEFAULT_SEASON_RANGE,
  SeasonRange,
  seasonPayload,
  type SeasonRangeValue,
} from '../ui/SeasonRange'

export function Search() {
  const [n, setN] = useState(16)
  const [maxTest, setMaxTest] = useState(2019)
  const [seed, setSeed] = useState(0)
  const [range, setRange] = useState<SeasonRangeValue>(DEFAULT_SEASON_RANGE)
  const [refresh, setRefresh] = useState(false)
  const [features, setFeatures] = useState<FeatureOverride>(NO_FEATURE_OVERRIDES)

  return (
    <JobRunner
      type="search"
      runLabel="Run search"
      buildPayload={() => ({
        n,
        max_test_season: maxTest,
        seed,
        ...seasonPayload(range),
        refresh,
        enable_features: features.enable,
        disable_features: features.disable,
      })}
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
          <div className="job-option">
            <label className="field-label" htmlFor="search-seed">Seed</label>
            <input
              id="search-seed"
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value) || 0)}
            />
          </div>
          <SeasonRange value={range} onChange={setRange} />
          <RefreshToggle value={refresh} onChange={setRefresh} />
          <FeatureToggles value={features} onChange={setFeatures} />
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
