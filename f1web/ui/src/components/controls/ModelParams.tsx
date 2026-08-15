import type { ReactNode } from 'react'
import './JobOptions.css'

/**
 * Editable HGB hyperparameter grid (the `[model.params]` keys from the config
 * schema). Used by the Train tab (per-run overrides, ephemeral) and the
 * Settings tab (locked display of the config values).
 */
export function ModelParams({
  keys,
  value,
  onChange,
  disabled = false,
  hint,
}: {
  /** Param keys in display order (model_params_keys from the config schema). */
  keys: string[]
  /** Current values; null renders empty fields (server falls back to config). */
  value: Record<string, number> | null
  /** Called with the full params dict on every edit. */
  onChange?: (value: Record<string, number>) => void
  /** Locked display mode (Settings tab): inputs greyed out, no edits. */
  disabled?: boolean
  hint?: ReactNode
}) {
  const params = value ?? {}
  return (
    <div className={disabled ? 'param-grid locked' : 'param-grid'}>
      {keys.map((key) => (
        <div className="field" key={key}>
          <label className="field-label" htmlFor={`param-${key}`}>
            {key}
          </label>
          <input
            id={`param-${key}`}
            type="number"
            step="any"
            disabled={disabled}
            value={params[key] !== undefined && Number.isFinite(params[key]) ? String(params[key]) : ''}
            onChange={(e) =>
              onChange?.({ ...params, [key]: e.target.value === '' ? 0 : Number(e.target.value) })
            }
          />
        </div>
      ))}
      {hint ? <p className="field-help">{hint}</p> : null}
    </div>
  )
}
