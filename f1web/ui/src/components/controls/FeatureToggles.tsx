import { getConfig } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { FeatureGroups } from './FeatureGroups'
import './JobOptions.css'

/** Per-run feature overrides, mirroring the CLI's --enable/--disable-features. */
export interface FeatureOverride {
  enable: string[]
  disable: string[]
}

/** No overrides: the job uses the config [features] enabled set as-is. */
export const NO_FEATURE_OVERRIDES: FeatureOverride = { enable: [], disable: [] }

/**
 * Per-run feature override control (CLI --enable-features / --disable-features):
 * every registered feature is shown as a checkbox whose checked state is the
 * *effective* enabled state for this run (config default, plus any force
 * on/off override). Nothing is written to config.toml.
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

  // Checkbox state = effective enabled for this run: an explicit force on/off
  // wins, otherwise the config default.
  const checked = (id: string): boolean => {
    if (value.enable.includes(id)) return true
    if (value.disable.includes(id)) return false
    return defaults.includes(id)
  }
  const toggle = (id: string, on: boolean) => {
    const enable = value.enable.filter((f) => f !== id)
    const disable = value.disable.filter((f) => f !== id)
    if (on === defaults.includes(id)) {
      // Matches the config default again -> drop the override.
      onChange({ enable, disable })
      return
    }
    if (on) enable.push(id)
    else disable.push(id)
    onChange({ enable, disable })
  }

  return (
    <FeatureGroups
      registry={registry}
      categories={categories}
      categoryMeta={category_meta}
      checked={checked}
      onToggle={toggle}
      resetLabel="Reset to config defaults"
      onReset={() => onChange(NO_FEATURE_OVERRIDES)}
    />
  )
}
