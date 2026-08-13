import { useEffect, useRef, useState } from 'react'
import { getBacktest, getModels, getStatus, type ModelsResponse } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { ErrorState, Skeleton } from '../ui/DataState'
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
import { BacktestRunResult } from './BacktestRunResult'
import { BacktestView } from './BacktestView'
import { ModelsOverview } from './ModelsOverview'
import { DEFAULT_MODEL, defaultStem, deployedName, modelChoices, selectedPaths } from './lib'
import './Backtest.css'

/**
 * Backtest: score every selected season with one or several saved models
 * (each using the features it was trained on) and compare them against the
 * grid / championship / zero baselines — the "how good is THIS model" view.
 * Two or more selected models additionally produce per-metric comparison
 * charts. Walk-forward retraining and output quantization live under
 * Advanced.
 */
export function Backtest() {
  const [checked, setChecked] = useState<string[]>([])
  const [walkForward, setWalkForward] = useState(false)
  const [quantize, setQuantize] = useState(true)
  const [range, setRange] = useState<SeasonRangeValue>(DEFAULT_SEASON_RANGE)
  const [refresh, setRefresh] = useState(false)
  const [features, setFeatures] = useState<FeatureOverride>(NO_FEATURE_OVERRIDES)
  const [version, setVersion] = useState(0)
  const status = useApi('status', () => getStatus())
  const modelsState = useApi<ModelsResponse>('models', getModels)
  const models = modelsState.state.phase === 'ready' ? modelsState.state.data : null
  const { state, retry } = useApi(`backtest-${version}`, () => getBacktest())
  const choices = modelChoices(models)
  const deployed = deployedName(models)
  // Guards the default selection: once the model index loads, preselect the
  // deployed model, but never clobber a selection the user made themselves.
  const userTouched = useRef(false)

  useEffect(() => {
    if (userTouched.current || models === null) return
    setChecked([deployed ?? DEFAULT_MODEL])
  }, [models, deployed])

  function toggleModel(value: string, on: boolean) {
    userTouched.current = true
    setChecked((current) => (on ? [...current, value] : current.filter((v) => v !== value)))
  }

  return (
    <>
      <ModelsOverview models={models} />
      <JobRunner
        type="backtest"
        runLabel="Run backtest"
        onDone={() => setVersion((v) => v + 1)}
        buildPayload={() => ({
          ...seasonPayload(range),
          refresh,
          quantize,
          // Walk-forward retraining is the explicit advanced opt-in; otherwise
          // the selected saved models are scored with their own features.
          use_checkpoint: !walkForward,
          ...(walkForward
            ? { enable_features: features.enable, disable_features: features.disable }
            : { model_paths: selectedPaths(models, checked) }),
        })}
        options={
          <>
            <div className="job-option model-select">
              <span className="job-label">Models to score</span>
              <div className="model-check-list">
                {choices.map((choice) => (
                  <label key={choice.value} className="check-line">
                    <input
                      type="checkbox"
                      checked={checked.includes(choice.value)}
                      disabled={walkForward}
                      onChange={(e) => toggleModel(choice.value, e.target.checked)}
                    />
                    {choice.label}
                  </label>
                ))}
              </div>
              <p className="job-option-hint">
                Each selected model scores the same seasons with its own features. Pick two
                or more to get comparison charts below; uncheck all to score the deployed
                checkpoint via the config feature set.
              </p>
            </div>
            <SeasonRange value={range} onChange={setRange} />
            <RefreshToggle value={refresh} onChange={setRefresh} />
            <details className="advanced-options">
              <summary>Advanced</summary>
              <div className="job-options-inner">
                <div className="job-option">
                  <label
                    className="check-line"
                    title="Ignore the selected models and retrain on every test season (train = all strictly earlier seasons)."
                  >
                    <input
                      type="checkbox"
                      checked={walkForward}
                      onChange={(e) => setWalkForward(e.target.checked)}
                    />
                    Walk-forward retraining
                  </label>
                  <p className="job-option-hint">
                    Honest out-of-sample estimates, but does not tell you how
                    good the model you just trained is.
                  </p>
                </div>
                <div className="job-option">
                  <label
                    className="check-line"
                    title="Round expected points to the nearest points-table value (matches the deployed predictor)."
                  >
                    <input
                      type="checkbox"
                      checked={quantize}
                      onChange={(e) => setQuantize(e.target.checked)}
                    />
                    Quantize points
                  </label>
                </div>
                {walkForward ? (
                  <FeatureToggles value={features} onChange={setFeatures} />
                ) : null}
              </div>
            </details>
          </>
        }
        renderResult={(job) => <BacktestRunResult job={job} />}
      />
      <PrereqHint
        when={status.state.phase === 'ready' && !status.state.data.model.has_checkpoint}
      >
        No model checkpoint yet — run Train first so there is a model to
        score.
      </PrereqHint>
      {state.phase === 'loading' ? (
        <Skeleton rows={8} />
      ) : state.phase === 'error' ? (
        <ErrorState message={state.message} onRetry={retry} />
      ) : (
        <BacktestView backtest={state.data} reference={defaultStem(models)} />
      )}
    </>
  )
}
