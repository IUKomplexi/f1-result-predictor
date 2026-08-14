import type { PredictionRow } from '../../api/client'
import { driverLabel } from '../../lib/format'
import { Badge } from '../ui/Badge'

/**
 * Editable qualifying grid for an upcoming race: one dropdown per driver
 * seeded from the prediction's own grid. The "Use model grid" option keeps
 * the model's guess for that driver; picking a number sets the grid position
 * override. Edits stay local until the user applies them (POST /api/predict
 * with a grid_csv body). Not shown for verified races — a completed race has
 * no grid to feed.
 */
export function GridEditor({
  drivers,
  values,
  onChange,
  onReset,
}: {
  drivers: PredictionRow[]
  /** null = pristine (dropdowns show the prediction's grid); edits populate it. */
  values: Record<string, string> | null
  onChange: (rows: Record<string, string>) => void
  onReset: () => void
}) {
  const isDirty = values !== null
  const positions = Array.from({ length: Math.max(drivers.length, 1) }, (_, i) => i + 1)
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
        Pick the real starting grid per driver. "Use model grid" keeps the
        model's guess.
      </p>
      <div className="grid-editor-table">
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
                    <select
                      className="grid-select"
                      value={current}
                      aria-label={`Grid position for ${driverLabel(row.driver_id)}`}
                      onChange={(e) =>
                        onChange({ ...(values ?? {}), [row.driver_id]: e.target.value })
                      }
                    >
                      <option value="">Use model grid</option>
                      {positions.map((p) => (
                        <option key={p} value={String(p)}>
                          {p}
                        </option>
                      ))}
                    </select>
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
