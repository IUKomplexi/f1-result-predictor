import './JobOptions.css'

/**
 * Re-fetch from the API instead of reusing cached data (CLI --refresh): when
 * checked the pipeline re-downloads raw responses from Jolpica, ignoring
 * everything in data/raw. Leave unchecked for the fast cached path; check it
 * after a race weekend (or an upstream data fix) to pick up new results.
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
        title="Re-download raw data from Jolpica instead of reusing the cached data/raw files (CLI --refresh)."
      >
        <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} />
        Re-fetch from API (ignore cache)
      </label>
      <p className="job-option-hint">
        Off by default: pipeline steps reuse the cached data in <code>data/raw</code>.
        Turn on to re-download from Jolpica — needed after a race weekend or an
        upstream data fix so new results are picked up.
      </p>
    </div>
  )
}
