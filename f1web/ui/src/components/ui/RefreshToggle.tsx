import './JobOptions.css'

/**
 * Refresh option for pipeline jobs: when checked, the job ignores its cache
 * and re-derives from the source of truth. Off by default (the fast cached
 * path). The default wording matches the CLI --refresh re-download semantic;
 * pages whose refresh clears a different cache (e.g. the prediction cache)
 * override the label/hint.
 */
export function RefreshToggle({
  value,
  onChange,
  label = 'Re-fetch from API (ignore cache)',
  hint = 'Off by default: saved data is reused. Turn on to download fresh data — do this after a race weekend so new results are picked up.',
}: {
  value: boolean
  onChange: (value: boolean) => void
  label?: string
  hint?: string
}) {
  return (
    <div className="job-option">
      <label className="refresh-check" title={label}>
        <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} />
        {label}
      </label>
      <p className="job-option-hint">{hint}</p>
    </div>
  )
}
