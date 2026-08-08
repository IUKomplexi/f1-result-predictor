import type { ReactNode } from 'react'

export function Skeleton({ rows = 4 }: { rows?: number }) {
  return (
    <div className="skeleton" aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="sk-card" />
      ))}
    </div>
  )
}

export function ErrorState({
  message,
  onRetry,
}: {
  message: string
  onRetry: () => void
}) {
  return (
    <section className="state-block" role="alert">
      <h2>Could not load this view</h2>
      <p>{message}</p>
      <button type="button" className="button" onClick={onRetry}>
        Retry
      </button>
    </section>
  )
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <section className="state-block">
      <h2>{title}</h2>
      {children ? <p>{children}</p> : null}
    </section>
  )
}

export function ProgressState({
  label,
  done,
  total,
}: {
  label: string
  done: number
  total: number
}) {
  const complete = total > 0 ? Math.round((done / total) * 100) : 0
  return (
    <section className="card progress-card" aria-busy="true">
      <p className="progress-label">{label}</p>
      <div
        className="progress-track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={total}
        aria-valuenow={done}
        aria-label={label}
      >
        <div className="progress-fill" style={{ width: `${complete}%` }} />
      </div>
    </section>
  )
}
