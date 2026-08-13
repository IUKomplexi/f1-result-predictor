import './JobOptions.css'

/**
 * Refresh raw-data cache toggle (CLI --refresh): when checked the pipeline
 * re-fetches from the API instead of reusing data/raw.
 */
export function RefreshToggle({
  value,
  onChange,
}: {
  value: boolean
  onChange: (value: boolean) => void
}) {
  return (
    <div className="job-option">
      <label
        className="refresh-check"
        title="Ignore the raw-data cache and re-fetch from the API (CLI --refresh)."
      >
        <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} />
        Refresh raw data (ignore cache)
      </label>
    </div>
  )
}
