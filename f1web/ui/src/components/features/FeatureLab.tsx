import { useState } from 'react'
import { ApiError, getConfig, getStatus, putConfig, type Job } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { fmtNumber } from '../../lib/format'
import { Badge } from '../ui/Badge'
import { JobRunner } from '../ui/JobRunner'
import { ModelPicker, modelCheckpointPath, useModels, type ModelSelection } from '../ui/ModelPicker'
import { PrereqHint } from '../ui/PrereqHint'
import { RefreshToggle } from '../ui/RefreshToggle'
import {
  DEFAULT_SEASON_RANGE,
  SeasonRange,
  seasonPayload,
  type SeasonRangeValue,
} from '../ui/SeasonRange'
import './FeatureLab.css'

/** "Use the config feature selection, not a saved model" pseudo-choice. */
const CONFIG_FEATURES = '__config__'

interface FeatureDelta {
  feature: string
  mae: number
  spearman: number
  delta_mae: number
  delta_spearman: number
}

/**
 * Feature Lab: "does this feature actually help?" — evaluates the current
 * feature set (config or a saved model's) on a walk-forward window, then
 * re-evaluates with each off feature added and each on feature removed.
 * Results show the ΔMAE/ΔSpearman; the winner can be applied to
 * features.enabled with one click.
 */
export function FeatureLab() {
  const status = useApi('status', () => getStatus())
  const { state: modelsState } = useModels()
  const models = modelsState.phase === 'ready' ? modelsState.data : null
  const [modelChoice, setModelChoice] = useState<ModelSelection>(CONFIG_FEATURES)
  const [range, setRange] = useState<SeasonRangeValue>(DEFAULT_SEASON_RANGE)
  const [refresh, setRefresh] = useState(false)
  const [applyStatus, setApplyStatus] = useState<string | null>(null)

  const applyChange = async (feature: string, action: 'add' | 'remove') => {
    setApplyStatus(null)
    try {
      const cfg = await getConfig()
      const current =
        (cfg.config.features?.enabled as string[] | null) ?? cfg.features.defaults
      const next =
        action === 'add'
          ? [...new Set([...current, feature])]
          : current.filter((f) => f !== feature)
      await putConfig({ ...cfg.config, features: { ...cfg.config.features, enabled: next } })
      setApplyStatus(
        `${action === 'add' ? 'Added' : 'Removed'} ${feature} in features.enabled — retrain (Train tab) for it to take effect.`,
      )
    } catch (error) {
      const message = error instanceof ApiError ? error.message : String(error)
      setApplyStatus(`Could not update features.enabled: ${message}`)
    }
  }

  return (
    <>
      <p className="muted config-intro">
        Answers <em>"does this feature actually help?"</em>. It evaluates the
        baseline feature set (the config selection, or a saved model's own set)
        on a walk-forward window, then re-evaluates once per candidate: every
        off feature added, every on feature removed. Deltas are vs the
        baseline — a negative <strong>ΔMAE</strong> means that variant scored
        better. Runs as a background job and can take a few minutes (one
        walk-forward per candidate).
      </p>
      <JobRunner
        type="features"
        runLabel="Evaluate features"
        buildPayload={() => ({
          ...seasonPayload(range),
          refresh,
          model_path:
            modelChoice === CONFIG_FEATURES
              ? null
              : modelCheckpointPath(models, modelChoice),
        })}
        options={
          <>
            <ModelPicker
              value={modelChoice}
              onChange={setModelChoice}
              extraOptions={[
                { value: CONFIG_FEATURES, label: 'config features (no model)' },
              ]}
              hint="Baseline feature set: a saved model's own features, or the config selection."
            />
            <SeasonRange value={range} onChange={setRange} />
            <RefreshToggle value={refresh} onChange={setRefresh} />
          </>
        }
        renderResult={(job) => (
          <FeatureResult job={job} onApply={applyChange} />
        )}
      />
      {applyStatus ? (
        <p className="save-status ok" role="status">{applyStatus}</p>
      ) : null}
      <PrereqHint
        when={status.state.phase === 'ready' && !status.state.data.data.has_dataset}
      >
        No dataset yet — run Train first so there are features to evaluate.
      </PrereqHint>
    </>
  )
}

function FeatureResult({
  job,
  onApply,
}: {
  job: Job
  onApply: (feature: string, action: 'add' | 'remove') => void
}) {
  const result = job.result ?? {}
  const baseline = (result.baseline ?? {}) as Record<string, number>
  const additions = (result.additions ?? []) as FeatureDelta[]
  const removals = (result.removals ?? []) as FeatureDelta[]
  return (
    <div className="result-block">
      <h3 className="card-title">Feature evaluation</h3>
      {job.log.length > 0 && (
        <details className="job-log">
          <summary>Log</summary>
          {job.log.map((line, i) => (
            <pre key={i} className="log-line">{line}</pre>
          ))}
        </details>
      )}
      <p className="summary-list">
        Baseline ({String(baseline.n_features)} features): MAE{' '}
        <code className="mono">{fmtNumber(baseline.mae, 4)}</code> · Spearman{' '}
        <code className="mono">{fmtNumber(baseline.spearman, 4)}</code>
      </p>
      <DeltaTable
        title="Adding a feature (negative Δ = better)"
        rows={additions}
        actionLabel="Add"
        action={(feature) => onApply(feature, 'add')}
      />
      <DeltaTable
        title="Removing a feature (negative Δ = removal helps)"
        rows={removals}
        actionLabel="Remove"
        action={(feature) => onApply(feature, 'remove')}
      />
    </div>
  )
}

function DeltaTable({
  title,
  rows,
  actionLabel,
  action,
}: {
  title: string
  rows: FeatureDelta[]
  actionLabel: string
  action: (feature: string) => void
}) {
  return (
    <section className="card feature-delta-card">
      <h4 className="card-title">{title}</h4>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Feature</th>
              <th scope="col" className="num">MAE</th>
              <th scope="col" className="num">Δ MAE</th>
              <th scope="col" className="num">Spearman</th>
              <th scope="col" className="num">Δ Spearman</th>
              <th scope="col" className="num">Verdict</th>
              <th scope="col" className="num">Apply</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const improved = row.delta_mae < -0.005
              const hurt = row.delta_mae > 0.005
              return (
                <tr key={row.feature}>
                  <td><code className="mono">{row.feature}</code></td>
                  <td className="num">{fmtNumber(row.mae, 4)}</td>
                  <td className="num">
                    <span className={row.delta_mae <= 0 ? 'delta-good' : 'delta-bad'}>
                      {fmtNumber(row.delta_mae, 4, true)}
                    </span>
                  </td>
                  <td className="num">{fmtNumber(row.spearman, 4)}</td>
                  <td className="num">
                    <span className={row.delta_spearman >= 0 ? 'delta-good' : 'delta-bad'}>
                      {fmtNumber(row.delta_spearman, 4, true)}
                    </span>
                  </td>
                  <td className="num">
                    {improved ? (
                      <Badge variant="ready">Helps</Badge>
                    ) : hurt ? (
                      <Badge variant="missing">Hurts</Badge>
                    ) : (
                      <Badge variant="warn">Neutral</Badge>
                    )}
                  </td>
                  <td className="num">
                    <button type="button" className="link-button" onClick={() => action(row.feature)}>
                      {actionLabel}
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}
