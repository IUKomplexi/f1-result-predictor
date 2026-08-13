import { useState } from 'react'
import { getConfig, getStatus, putConfig, type Job } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { FeatureToggles, NO_FEATURE_OVERRIDES, type FeatureOverride } from '../ui/FeatureToggles'
import { JobRunner } from '../ui/JobRunner'
import { PrereqHint } from '../ui/PrereqHint'
import { RefreshToggle } from '../ui/RefreshToggle'
import {
  DEFAULT_SEASON_RANGE,
  SeasonRange,
  seasonPayload,
  type SeasonRangeValue,
} from '../ui/SeasonRange'

/**
 * Hyperparameter search: samples HGB configurations, evaluates each on a
 * reduced walk-forward window, and ranks them. The winner is written to
 * [model.params] (Settings) — it tunes *hyperparameters*, not features.
 * Feature selection happens via the toggles here or the Feature Lab tab.
 */
export function Search() {
  const status = useApi('status', () => getStatus())
  const [n, setN] = useState(16)
  const [maxTest, setMaxTest] = useState(2019)
  const [seed, setSeed] = useState(0)
  const [range, setRange] = useState<SeasonRangeValue>(DEFAULT_SEASON_RANGE)
  const [refresh, setRefresh] = useState(false)
  const [features, setFeatures] = useState<FeatureOverride>(NO_FEATURE_OVERRIDES)

  return (
    <>
      <p className="muted config-intro">
        Tunes the model's <strong>hyperparameters</strong> ([model.params]:
        max_iter, learning_rate, max_depth, …). It samples{' '}
        <code>n</code> random configurations, evaluates each with a fast
        reduced-window walk-forward, and ranks them — the best one can be
        applied to the config with one click. This is <em>not</em> a feature
        search: the enabled features below are fixed for every candidate. After
        applying a result, retrain the model (Train tab).
      </p>
      <JobRunner
        type="search"
        runLabel="Run search"
        buildPayload={() => {
          const err = validateSearch(n, maxTest, range, status.state)
          if (err) throw new Error(err)
          return {
            n,
            max_test_season: maxTest,
            seed,
            ...seasonPayload(range),
            refresh,
            enable_features: features.enable,
            disable_features: features.disable,
          }
        }}
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
              <p className="job-option-hint">Number of random configs to sample.</p>
            </div>
            <div className="job-option">
              <label className="field-label" htmlFor="max-test">Max test season</label>
              <input
                id="max-test"
                type="number"
                value={maxTest}
                onChange={(e) => setMaxTest(Number(e.target.value) || 2019)}
              />
              <p className="job-option-hint">
                Latest season used for evaluation; every earlier season is
                training data. Needs at least 3 seasons before it to train on.
              </p>
            </div>
            <div className="job-option">
              <label className="field-label" htmlFor="search-seed">Seed</label>
              <input
                id="search-seed"
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value) || 0)}
              />
              <p className="job-option-hint">Random seed — same seed, same sampled configs.</p>
            </div>
            <SeasonRange value={range} onChange={setRange} />
            <RefreshToggle value={refresh} onChange={setRefresh} />
            <FeatureToggles value={features} onChange={setFeatures} />
          </>
        }
        renderResult={(job) => <SearchResult job={job} />}
      />
      <PrereqHint
        when={status.state.phase === 'ready' && !status.state.data.data.has_dataset}
      >
        No dataset yet — run Train first so the search has features to tune
        against.
      </PrereqHint>
    </>
  )
}

/**
 * Validate the search inputs up front so a bad configuration fails fast with
 * an actionable message instead of an opaque mid-run error. Returns an error
 * string, or null when the inputs are usable.
 */
function validateSearch(
  n: number,
  maxTest: number,
  range: SeasonRangeValue,
  status: { phase: string; data?: { seasons: { start: number; end: number; data_end: number } } },
): string | null {
  if (n < 1) return 'Search n must be at least 1.'
  if (status.phase !== 'ready') return null
  const { start: cfgStart, end: cfgEnd, data_end: dataEnd } = status.data!.seasons
  const start = range.start ?? cfgStart
  const end = range.end ?? cfgEnd
  const seasons = end - start + 1
  if (seasons < 4) {
    return `Search needs at least 4 seasons (3 to train on + 1 to test); ${start}–${end} is only ${seasons}.`
  }
  if (maxTest > end) {
    return `Max test season ${maxTest} is after the search range end (${end}).`
  }
  if (maxTest < start + 3) {
    return `Max test season ${maxTest} leaves fewer than 3 earlier seasons to train on (start ${start}).`
  }
  if (maxTest > dataEnd) {
    return `Max test season ${maxTest} has no data yet (latest fetched season: ${dataEnd}). Fetch it on the Data tab first.`
  }
  return null
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
      <p className="job-option-hint">
        Applied params only take effect after retraining (Train tab).
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
