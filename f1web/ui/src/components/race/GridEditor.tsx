import type { PredictionRow } from '../../api/client'
import { driverLabel } from '../../lib/format'
import { Badge } from '../ui/Badge'

/**
 * Editable qualifying grid for an upcoming race: one numeric input per
 * driver, seeded from the prediction's own grid. Edits stay local until the
 * user applies them (POST /api/predict with a grid_csv body); empty cells
 * keep the model's grid for that driver. Not shown for verified races — a
 * completed race has no grid to feed.
 */
export function GridEditor({
  drivers,
  values,
  onChange,
  onReset,
}: {
  drivers: PredictionRow[]
  /** null = pristine (inputs show the prediction's grid); edits populate it. */
  values: Record<string, string> | null
  onChange: (rows: Record<string, string>) => void
  onReset: () => void
}) {
  const isDirty = values !== null
  const invalid = (value: string) => value.trim() !== '' && !/^\d+$/.test(value.trim())
  return (
    <div className="job-option grid-editor">
      <div className="feature-toggle-head">
        <span className="job-label">
          Qualifying grid override (CLI --grid)
          {isDirty ? <Badge variant="warn">pending</Badge> : null}
        </span>
        <button type="button" className="link-button" onClick={onReset} disabled={!isDirty}>
          Reset to model grid
        </button>
      </div>
      <p className="job-option-hint">
        Feed the actual qualifying grid before applying — positions take effect
        on the next prediction. Empty cells keep the model's grid for that driver.
      </p>
      <div className="table-wrap grid-editor-table">
        <table className="data-table grid-table">
          <thead>
            <tr>
              <th scope="col">Driver</th>
              <th scope="col" className="num">Grid</th>
            </tr>
          </thead>
          <tbody>
            {drivers.map((row) => {
              const current =
                values !== null
                  ? (values[row.driver_id] ?? String(row.grid ?? ''))
                  : String(row.grid ?? '')
              return (
                <tr key={row.driver_id}>
                  <td className="driver">{driverLabel(row.driver_id)}</td>
                  <td className="num">
                    <input
                      type="text"
                      inputMode="numeric"
                      className={`grid-input${invalid(current) ? ' invalid' : ''}`}
                      value={current}
                      aria-label={`Grid position for ${driverLabel(row.driver_id)}`}
                      aria-invalid={invalid(current)}
                      onChange={(e) =>
                        onChange({ ...(values ?? {}), [row.driver_id]: e.target.value })
                      }
                    />
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
