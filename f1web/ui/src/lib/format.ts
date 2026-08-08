/** Human-readable formatting for dashboard values. */

export function driverLabel(id: string | null | undefined): string {
  if (!id) return '–'
  return id
    .replace(/_/g, ' ')
    .replace(/\b[a-z]/g, (c) => c.toUpperCase())
}

export function fmtNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '–'
  return value.toFixed(digits)
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
