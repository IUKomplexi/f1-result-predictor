/** Human-readable formatting for dashboard values. */

export function driverLabel(id: string | null | undefined): string {
  if (!id) return '–'
  return id
    .replace(/_/g, ' ')
    .replace(/\b[a-z]/g, (c) => c.toUpperCase())
}

export function fmtNumber(value: number | null | undefined, digits = 2, sign = false): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '–'
  const text = value.toFixed(digits)
  return sign && value > 0 ? `+${text}` : text
}

export function fmtPoints(value: number | null | undefined): string {
  return fmtNumber(value, 1)
}

export function pct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '–'
  return `${Math.round(value * 100)}%`
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return '–'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toISOString().slice(0, 10)
}

/** Seconds -> "45s", "2m 05s", "1h 03m" (monospace-friendly). */
export function fmtElapsed(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return '–'
  const total = Math.max(0, Math.round(seconds))
  if (total < 60) return `${total}s`
  const minutes = Math.floor(total / 60)
  if (minutes < 60) return `${minutes}m ${String(total % 60).padStart(2, '0')}s`
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`
}
