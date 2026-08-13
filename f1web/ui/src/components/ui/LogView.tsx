import { useEffect, useRef } from 'react'

/**
 * Auto-scrolling terminal-style log block. Stays pinned to the newest line
 * while a job streams output; reuse for the job queue panel and JobRunner.
 */
export function LogView({ lines, maxHeight = '16rem' }: { lines: string[]; maxHeight?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = ref.current
    if (el) el.scrollTop = el.scrollHeight
  }, [lines])
  return (
    <div className="log-view" ref={ref} style={{ maxHeight }}>
      {lines.length === 0 ? (
        <p className="muted log-empty">No output yet…</p>
      ) : (
        lines.map((line, i) => (
          <pre key={i} className="log-line">{line}</pre>
        ))
      )}
    </div>
  )
}
