import type { ModelsResponse } from '../../api/client'
import { modelChoices } from '../backtest/lib'
import { RACE_DEFAULT_MODEL } from './lib'

/**
 * Checkpoint selector for a single race (CLI --model): pick a saved model or
 * the config-default checkpoint. Named models keep their own calibrators
 * (sibling <name>.calibrators.joblib, loaded by the backend).
 */
export function ModelPicker({
  models,
  value,
  onChange,
}: {
  models: ModelsResponse | null
  value: string
  onChange: (value: string) => void
}) {
  if (models === null) {
    return (
      <div className="field">
        <label className="field-label" htmlFor="race-model">
          Model
        </label>
        <select id="race-model" className="select" disabled>
          <option>Loading models…</option>
        </select>
      </div>
    )
  }
  return (
    <div className="field">
      <label className="field-label" htmlFor="race-model">
        Model
      </label>
      <select
        id="race-model"
        className="select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {modelChoices(models).map((choice) => (
          <option key={choice.value} value={choice.value}>
            {choice.label}
          </option>
        ))}
      </select>
      <p className="job-option-hint">
        The selected checkpoint scores this race, with its own calibrators when it has them.
      </p>
    </div>
  )
}

export { RACE_DEFAULT_MODEL }
