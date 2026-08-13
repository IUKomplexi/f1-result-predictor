import { getConfig } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import './JobOptions.css'

/** Per-run feature overrides, mirroring the CLI's --enable/--disable-features. */
export interface FeatureOverride {
  enable: string[]
  disable: string[]
}

/** No overrides: the job uses the config [features] enabled set as-is. */
export const NO_FEATURE_OVERRIDES: FeatureOverride = { enable: [], disable: [] }

type Mode = 'default' | 'enable' | 'disable'

/**
 * Per-run feature override control (CLI --enable-features / --disable-features):
 * every registered feature can stay at its config default or be forced on/off
 * for this run only. Nothing is written to config.toml.
 */
export function FeatureToggles({
  value,
  onChange,
}: {
  value: FeatureOverride
  onChange: (value: FeatureOverride) => void
}) {
  const { state } = useApi('feature-toggle-meta', () => getConfig())
  if (state.phase === 'loading') {
    return (
      <div className="job-option feature-toggles">
        <span className="job-label">Feature overrides</span>
        <p className="muted">Loading feature registry…</p>
      </div>
    )
  }
  if (state.phase === 'error') {
    return (
      <div className="job-option feature-toggles">
        <span className="job-label">Feature overrides</span>
        <p className="job-options-error">Could not load features: {state.message}</p>
      </div>
    )
  }
  const { registry, categories, defaults, category_meta } = state.data.features
  // Groups come from the backend (features/registry.py CATEGORY_ORDER +
  // CATEGORY_LABELS via /api/config); categories unknown to the backend still
  // render, appended after the known ones so drift never hides a feature.
  const known = new Set(category_meta.map((m) => m.id))
  const groups = [
    ...category_meta,
    ...[...new Set(Object.values(categories))]
      .filter((category) => !known.has(category))
      .map((category) => ({ id: category, label: category })),
  ]
  const modeOf = (id: string): Mode =>
    value.enable.includes(id) ? 'enable' : value.disable.includes(id) ? 'disable' : 'default'
  const setMode = (id: string, mode: Mode) => {
    const enable = value.enable.filter((f) => f !== id)
    const disable = value.disable.filter((f) => f !== id)
    if (mode === 'enable') enable.push(id)
    if (mode === 'disable') disable.push(id)
    onChange({ enable, disable })
  }
  return (
    <div className="job-option feature-toggles">
      <div className="feature-toggle-head">
        <span className="job-label">Feature overrides</span>
        <button
          type="button"
          className="link-button"
          onClick={() => onChange(NO_FEATURE_OVERRIDES)}
        >
          Reset to config defaults
        </button>
      </div>
      <div className="feature-toggle-groups">
        {groups.map((group) => {
          const ids = registry.filter((id) => categories[id] === group.id)
          if (ids.length === 0) return null
          return (
            <div key={group.id} className="feature-toggle-group">
              <h4 className="feature-toggle-group-title">{group.label}</h4>
              {ids.map((id) => (
                <label key={id} className="feature-toggle-row">
                  <span className="mono">{id}</span>
                  <select
                    className="feature-toggle-select"
                    value={modeOf(id)}
                    onChange={(e) => setMode(id, e.target.value as Mode)}
                    aria-label={`${id} feature override`}
                  >
                    <option value="default">
                      {defaults.includes(id) ? 'default (on)' : 'default (off)'}
                    </option>
                    <option value="enable">force on</option>
                    <option value="disable">force off</option>
                  </select>
                </label>
              ))}
            </div>
          )
        })}
      </div>
    </div>
  )
}
