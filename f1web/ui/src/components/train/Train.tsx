import { useState } from 'react'
import { getStatus, type Job } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { JobRunner } from '../ui/JobRunner'
import { PrereqHint } from '../ui/PrereqHint'
import { FeatureToggles, NO_FEATURE_OVERRIDES, type FeatureOverride } from '../ui/FeatureToggles'
import {
  DEFAULT_SEASON_RANGE,
  SeasonRange,
  seasonPayload,
  type SeasonRangeValue,
} from '../ui/SeasonRange'
import { RefreshToggle } from '../ui/RefreshToggle'

export function Train() {
  const status = useApi('status', () => getStatus())
  const [range, setRange] = useState<SeasonRangeValue>(DEFAULT_SEASON_RANGE)
  const [refresh, setRefresh] = useState(false)
  const [features, setFeatures] = useState<FeatureOverride>(NO_FEATURE_OVERRIDES)
  const [name, setName] = useState('')
  const suggestion =
    range.start !== null && range.end !== null ? `hurdle-${range.start}-${range.end}` : 'hurdle'
  return (
    <>
      <JobRunner
        type="train"
        runLabel="Train model"
      buildPayload={() => ({
        ...seasonPayload(range),
        refresh,
        ...(name.trim() !== '' ? { name: name.trim() } : {}),
        enable_features: features.enable,
        disable_features: features.disable,
      })}
      options={
        <>
          <SeasonRange value={range} onChange={setRange} />
          <div className="job-option">
            <label className="field-label" htmlFor="train-name">Model name</label>
            <input
              id="train-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={suggestion}
            />
            <p className="job-option-hint">
              Blank overwrites the configured checkpoint; a name saves a separate{' '}
              <code>data/model/&lt;name&gt;.joblib</code> you can pick on Specific Race.
            </p>
          </div>
          <RefreshToggle value={refresh} onChange={setRefresh} />
          <FeatureToggles value={features} onChange={setFeatures} />
        </>
      }
      renderResult={(job) => <TrainResult job={job} />}
      />
      <PrereqHint
        when={status.state.phase === 'ready' && !status.state.data.data.has_raw_cache}
      >
        No raw data cached yet — run Fetch data on the Data tab first, or Train will
        pull from the API itself (slow).
      </PrereqHint>
    </>
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
