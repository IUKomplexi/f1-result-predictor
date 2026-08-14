import { useState } from 'react'
import { getStatus, type Job } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { JobRunner } from '../ui/JobRunner'
import { PrereqHint } from '../ui/PrereqHint'
import { FeatureToggles, NO_FEATURE_OVERRIDES, type FeatureOverride } from '../ui/FeatureToggles'
import {
  DEFAULT_SEASON_RANGE,
  SeasonRange,
  resolveRange,
  seasonPayload,
  seasonRangeError,
  type SeasonRangeValue,
} from '../ui/SeasonRange'

export function Train() {
  const status = useApi('status', () => getStatus())
  const [range, setRange] = useState<SeasonRangeValue>(DEFAULT_SEASON_RANGE)
  const [features, setFeatures] = useState<FeatureOverride>(NO_FEATURE_OVERRIDES)
  const [name, setName] = useState('')
  const suggestion =
    range.start !== null && range.end !== null ? `hurdle-${range.start}-${range.end}` : 'hurdle'
  const seasons = status.state.phase === 'ready' ? status.state.data.seasons : null
  return (
    <>
      <JobRunner
        type="train"
        runLabel="Train model"
      buildPayload={() => {
        const resolved = resolveRange(range, seasons)
        const rangeError = seasonRangeError(resolved)
        if (rangeError) throw new Error(rangeError)
        return {
          ...seasonPayload(resolved),
          ...(name.trim() !== '' ? { name: name.trim() } : {}),
          enable_features: features.enable,
          disable_features: features.disable,
        }
      }}
      options={
        <>
          <p className="job-option-hint">
            Training also calibrates the model automatically — no separate
            calibrate step needed.
          </p>
          <SeasonRange value={range} onChange={setRange} />
          <div className="job-option">
            <label className="field-label" htmlFor="train-name">Model name</label>
            <input
              id="train-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value.replace(/[^A-Za-z0-9._-]/g, ''))}
              pattern="[A-Za-z0-9._-]*"
              placeholder={suggestion}
            />
            <p className="job-option-hint">
              Leave empty to replace the current model.
            </p>
          </div>
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
  const named = (result.checkpoint as string | undefined) ?? ''
  const stem = named.split(/[\\/]/).pop()?.replace(/\.joblib$/, '')
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
        <li>checkpoint: <code className="mono">{named}</code></li>
      </ul>
      <p className="job-option-hint">
        Done. To see how good it is, open the{' '}
        <strong>Backtest</strong> tab and pick{' '}
        {stem ? <code className="mono">{stem}</code> : 'the default model'}.
      </p>
    </div>
  )
}
