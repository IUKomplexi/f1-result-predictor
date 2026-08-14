import type { PredictionRow } from '../../api/client'
import { driverLabel } from '../../lib/format'
import { Badge } from '../ui/Badge'

/**
 * Editable qualifying grid for an upcoming race: one row per starting
 * position, with a driver dropdown to assign who starts there. Seeded from
 * the prediction's own grid. "Use model grid" keeps the model's guess for
 * that position. Picking a driver clears them from any other position
 * (swap semantics), so the same driver can never occupy two rows. Edits
 * stay local until the user applies them (POST /api/predict with a
 * grid_csv body). Not shown for verified races — a completed race has no
 * grid to feed.
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

  // The driver currently assigned to a position: an explicit override wins,
  // otherwise the prediction's own grid.
  const driverAt = (position: number): string => {
    if (values !== null) {
      for (const [driverId, pos] of Object.entries(values)) {
        if (pos === String(position)) return driverId
      }
    }
    return drivers.find((row) => row.grid === position)?.driver_id ?? ''
  }

  const selectDriver = (position: number, driverId: string) => {
    const next = { ...(values ?? {}) }
    // Clearing: remove whoever was assigned to this position.
    for (const [id, pos] of Object.entries(next)) {
      if (pos === String(position)) delete next[id]
    }
    if (driverId === '') {
      onChange(next)
      return
    }
    // The chosen driver moves here from wherever they were before.
    for (const [id, pos] of Object.entries(next)) {
      if (id === driverId && pos !== String(position)) delete next[id]
    }
    next[driverId] = String(position)
    onChange(next)
  }

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
        Assign a driver to each starting position. "Use model grid" keeps the
        model's guess for that position.
      </p>
      <div className="grid-editor-table">
        <table className="data-table grid-table">
          <thead>
            <tr>
              <th scope="col" className="num">Pos</th>
              <th scope="col">Driver</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((position) => {
              const current = driverAt(position)
              return (
                <tr key={position}>
                  <td className="num rank">{position}</td>
                  <td>
                    <select
                      className="grid-select"
                      value={current}
                      aria-label={`Driver at position ${position}`}
                      onChange={(e) => selectDriver(position, e.target.value)}
                    >
                      <option value="">Use model grid</option>
                      {drivers.map((row) => (
                        <option key={row.driver_id} value={row.driver_id}>
                          {driverLabel(row.driver_id)}
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
