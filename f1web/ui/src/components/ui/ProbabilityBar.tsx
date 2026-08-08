import { pct } from '../../lib/format'

/** A labeled horizontal probability bar (used for p_scored / p_top3 / p_win). */
export function ProbabilityBar({
  value,
  label,
  variant,
}: {
  value: number | null | undefined
  label: string
  variant: 'scored' | 'top3' | 'win'
}) {
  const clamped = Math.min(1, Math.max(0, value ?? 0))
  return (
    <span className="prob" title={`${label}: ${pct(value)}`}>
      <span className={`prob-track ${variant}`}>
        <span
          className="prob-fill"
          style={{ width: `${Math.round(clamped * 100)}%` }}
        />
      </span>
      <span className="prob-label">{pct(value)}</span>
    </span>
  )
}
