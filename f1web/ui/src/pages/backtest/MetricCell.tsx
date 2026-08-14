import type { BacktestMetricRow } from '../../api/client'
import { fmtNumber } from '../../lib/format'
import { cellTone } from './lib'

/**
 * A color-coded metric cell in a comparison table: best value in the column
 * is green, worst red, strictly-between values orange, ties white.
 */
export function MetricCell({
  metric,
  values,
  value,
}: {
  metric: keyof BacktestMetricRow
  /** Every other row's value for this metric column (for best/worst). */
  values: number[]
  value: number | null | undefined
}) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return <td className="num muted">–</td>
  }
  return (
    <td className={`num cell-${cellTone(metric, values, value)}`}>
      {fmtNumber(value, 3)}
    </td>
  )
}
