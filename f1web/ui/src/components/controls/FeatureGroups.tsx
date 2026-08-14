import type { ReactNode } from 'react'
import './JobOptions.css'

/** Category display metadata from /api/config (id + label, display order). */
export interface CategoryMeta {
  id: string
  label: string
}

/**
 * Shared grouped feature-selection control: the Train-tab arrangement
 * (grouped cards, one row per feature) with a checkbox per feature. Used by
 * Settings (config-level selection) and Train (per-run overrides) so both
 * look and behave identically.
 */
export function FeatureGroups({
  registry,
  categories,
  categoryMeta,
  checked,
  onToggle,
  resetLabel,
  onReset,
  hint,
}: {
  /** Every feature id, in registry order. */
  registry: string[]
  /** feature id -> category id. */
  categories: Record<string, string>
  /** Category display order + labels. */
  categoryMeta: CategoryMeta[]
  /** Effective enabled state of a feature (the checkbox's checked value). */
  checked: (id: string) => boolean
  /** Toggle a feature's effective state. */
  onToggle: (id: string, checked: boolean) => void
  /** Reset action (e.g. 'Reset to config defaults'). */
  resetLabel: string
  onReset: () => void
  /** Optional extra content below the groups (help text etc.). */
  hint?: ReactNode
}) {
  // Groups come from the backend (features/registry.py CATEGORIES +
  // CATEGORY_LABELS via /api/config); categories unknown to the backend still
  // render, appended after the known ones so drift never hides a feature.
  const known = new Set(categoryMeta.map((meta) => meta.id))
  const groups = [
    ...categoryMeta,
    ...[...new Set(Object.values(categories))]
      .filter((category) => !known.has(category))
      .map((category) => ({ id: category, label: category })),
  ]
  return (
    <div className="job-option feature-toggles">
      <div className="feature-toggle-head">
        <span className="job-label">Features</span>
        <button type="button" className="link-button" onClick={onReset}>
          {resetLabel}
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
                  <input
                    type="checkbox"
                    className="feature-toggle-check"
                    checked={checked(id)}
                    onChange={(e) => onToggle(id, e.target.checked)}
                    aria-label={`${id} feature`}
                  />
                </label>
              ))}
            </div>
          )
        })}
      </div>
      {hint}
    </div>
  )
}
