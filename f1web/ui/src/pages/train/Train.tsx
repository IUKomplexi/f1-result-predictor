import { useEffect, useRef, useState } from 'react'
import { getConfig, getModels, getStatus, type Job } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { JobRunner } from '../../components/jobs/JobRunner'
import { PrereqHint } from '../../components/ui/PrereqHint'
import { FeatureToggles, NO_FEATURE_OVERRIDES, type FeatureOverride } from '../../components/controls/FeatureToggles'
import { ModelParams } from '../../components/controls/ModelParams'
import { deployedName, formatParams } from '../../lib/models'
import {
  DEFAULT_SEASON_RANGE,
  SeasonRange,
  resolveRange,
  seasonPayload,
  seasonRangeError,
  type SeasonRangeValue,
} from '../../components/controls/SeasonRange'

export function Train() {
  const status = useApi('status', () => getStatus())
  const config = useApi('config', () => getConfig())
  const models = useApi('models', () => getModels())
  const [range, setRange] = useState<SeasonRangeValue>(DEFAULT_SEASON_RANGE)
  const [features, setFeatures] = useState<FeatureOverride>(NO_FEATURE_OVERRIDES)
  const [name, setName] = useState('')
  const [params, setParams] = useState<Record<string, number> | null>(null)
  const prefilled = useRef(false)
  const edited = useRef(false)
  const suggestion =
    range.start !== null && range.end !== null ? `hurdle-${range.start}-${range.end}` : 'hurdle'
  const seasons = status.state.phase === 'ready' ? status.state.data.seasons : null
  const rangeError =
    seasons !== null ? seasonRangeError(resolveRange(range, seasons)) : null

  // Prefill the hyperparameter editor with the deployed model's trained-on
  // params (what you see is what the current model uses), falling back to the
  // config values; never clobbers edits the user already made. Waits for both
  // fetches so the deployed model always wins over the config fallback.
  const handleParamsChange = (next: Record<string, number>) => {
    edited.current = true
    setParams(next)
  }
  useEffect(() => {
    if (edited.current || prefilled.current || params !== null) return
    if (models.state.phase !== 'ready' || config.state.phase !== 'ready') return
    const deployed = deployedName(models.state.data)
    const trained = deployed !== null ? models.state.data.models[deployed]?.params : undefined
    if (trained !== undefined && Object.keys(trained).length > 0) {
      setParams(trained)
    } else {
      const cfgParams = config.state.data.config.model?.params as Record<string, number> | undefined
      if (cfgParams !== undefined && Object.keys(cfgParams).length > 0) {
        setParams(cfgParams)
      }
    }
    prefilled.current = true
  }, [models.state.phase, config.state.phase, params])

  return (
    <>
      <JobRunner
        type="train"
        runLabel="Train model"
        buttonAlign="end"
        layout="train"
        buildPayload={() => {
          const resolved = resolveRange(range, seasons)
          const rangeError = seasonRangeError(resolved)
          if (rangeError) throw new Error(rangeError)
          return {
            ...seasonPayload(resolved),
            ...(name.trim() !== '' ? { name: name.trim() } : {}),
            ...(params !== null ? { params } : {}),
            enable_features: features.enable,
            disable_features: features.disable,
          }
        }}
        options={
          <>
            <div className="job-option train-model-option">
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
                Training also calibrates the model automatically — no separate calibrate step needed.
              </p>
            </div>
            <div className="season-pair">
              <SeasonRange value={range} onChange={setRange} />
              {rangeError !== null ? (
                <p className="save-status error season-range-error" role="alert">
                  {rangeError}
                </p>
              ) : null}
            </div>
            <div className="job-option train-params">
              <span className="job-label">Hyperparameters</span>
              <ModelParams
                keys={config.state.phase === 'ready' ? config.state.data.model_params_keys : []}
                value={params}
                onChange={handleParamsChange}
                hint="Defaults to the deployed model's trained-on params. Per-run only — config.toml is untouched."
              />
            </div>
            <div className="train-features">
              <FeatureToggles value={features} onChange={setFeatures} />
            </div>
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
  const paramsText = formatParams(result.params as Record<string, number> | undefined)
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
        {paramsText !== null ? (
          <li>params: <code className="mono">{paramsText}</code></li>
        ) : null}
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
