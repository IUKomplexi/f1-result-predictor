import { getModels, type ModelsResponse } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import './JobOptions.css'

export type ModelSelection = 'default' | string

/**
 * Loaded saved-model index (``GET /api/models``). ``state.data.models`` maps
 * checkpoint stems to metadata (features, seasons, params, …).
 */
export function useModels() {
  return useApi<ModelsResponse>('models', getModels)
}

/** Resolve a ModelSelection to a checkpoint path (null = no model override). */
export function modelCheckpointPath(
  models: ModelsResponse | null,
  selection: ModelSelection,
): string | null {
  if (selection === 'default') {
    return models?.default ?? null
  }
  return models?.models[selection]?.checkpoint ?? null
}

/**
 * Saved-model selector for pipeline jobs (Backtest, Calibration, Feature
 * Lab). Choosing a model makes the job run against that checkpoint with the
 * features it was trained on — no separate feature selection needed.
 */
export function ModelPicker({
  value,
  onChange,
  disabled,
  hint,
  extraOptions = [],
}: {
  value: ModelSelection
  onChange: (value: ModelSelection) => void
  disabled?: boolean
  hint?: string
  /** Extra choices prepended to the list (e.g. "config features, no model"). */
  extraOptions?: { value: string; label: string }[]
}) {
  const { state } = useModels()
  const models = state.phase === 'ready' ? state.data : null
  const names = models ? Object.keys(models.models).sort() : []
  const selected = value !== 'default' ? models?.models[value] : null
  const selectedFeatures = selected?.features
  return (
    <div className="job-option">
      <label className="job-label" htmlFor="model-picker">
        Model
      </label>
      <select
        id="model-picker"
        className="select"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="default">config default (deployed)</option>
        {extraOptions.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
        {names.map((name) => (
          <option key={name} value={name}>
            {name}
          </option>
        ))}
      </select>
      {selectedFeatures && value !== 'default' ? (
        <p className="job-option-hint">
          Uses the {selectedFeatures.length} features this model was trained on:{' '}
          {selectedFeatures.join(', ')}
        </p>
      ) : hint ? (
        <p className="job-option-hint">{hint}</p>
      ) : (
        <p className="job-option-hint">
          The job runs against this checkpoint with the features it was trained
          on — no separate feature selection needed.
        </p>
      )}
    </div>
  )
}
