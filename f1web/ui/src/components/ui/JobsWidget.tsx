import { useEffect, useRef, useState } from 'react'
import { type JobSummary } from '../../api/client'
import { useJob, useJobs } from '../../hooks/useJob'
import { fmtElapsed } from '../../lib/format'
import { Badge } from './Badge'
import { LogView } from './LogView'
import './JobsWidget.css'

const STATUS_VARIANT: Record<JobSummary['status'], 'ready' | 'missing' | 'warn' | 'info'> = {
  queued: 'warn',
  running: 'info',
  done: 'ready',
  failed: 'missing',
}

const STATUS_LABEL: Record<JobSummary['status'], string> = {
  queued: 'Queued',
  running: 'Running',
  done: 'Done',
  failed: 'Failed',
}

/**
 * Header job-queue widget: a compact trigger showing the running/queued job
 * count, and a dropdown panel with the full queue, per-job status + elapsed
 * time, and an auto-scrolling log with copy-to-clipboard for the selected job.
 * Visible on every tab so pipeline activity never happens invisibly.
 */
export function JobsWidget() {
  const { jobs } = useJobs()
  const [open, setOpen] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [now, setNow] = useState(() => Date.now())
  const widgetRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const [announcement, setAnnouncement] = useState('')
  const prevStatuses = useRef<Record<string, JobSummary['status']>>({})

  const running = jobs.find((j) => j.status === 'running') ?? null
  const activeCount = jobs.filter((j) => j.status === 'running' || j.status === 'queued').length
  const selected = selectedId ?? running?.id ?? null
  const job = useJob(selected)
  const elapsed =
    running !== null && running.started_at !== null
      ? (now - running.started_at * 1000) / 1000
      : 0

  // Tick once per second while something is running (or the panel is open) so
  // elapsed times move.
  useEffect(() => {
    if (!running && !open) return
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [running?.id, open])

  // Announce status transitions (started/finished/failed) to screen readers.
  useEffect(() => {
    for (const entry of jobs) {
      const prev = prevStatuses.current[entry.id]
      if (prev !== undefined && prev !== entry.status) {
        setAnnouncement(`${entry.label}: ${STATUS_LABEL[entry.status]}`)
      }
      prevStatuses.current[entry.id] = entry.status
    }
  }, [jobs])

  // Dialog behavior while open: focus the panel, Escape closes (and restores
  // focus to the trigger), and a click outside closes the panel.
  useEffect(() => {
    if (!open) return
    panelRef.current?.focus()
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        event.preventDefault()
        closePanel()
      }
    }
    function onMouseDown(event: MouseEvent) {
      if (
        widgetRef.current &&
        event.target instanceof Node &&
        !widgetRef.current.contains(event.target)
      ) {
        setOpen(false)
      }
    }
    document.addEventListener('keydown', onKeyDown)
    document.addEventListener('mousedown', onMouseDown)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      document.removeEventListener('mousedown', onMouseDown)
    }
  }, [open])

  function closePanel() {
    setOpen(false)
    triggerRef.current?.focus()
  }

  const copyLog = async () => {
    if (!job) return
    try {
      await navigator.clipboard.writeText(job.log.join('\n'))
    } catch {
      // Clipboard unavailable (permissions, http): nothing useful to do.
    }
  }

  return (
    <div className="jobs-widget" ref={widgetRef}>
      <span className="sr-only" aria-live="polite">
        {announcement}
      </span>
      {running ? (
        <span className="jobs-running-pill">
          Running: {running.label} · {fmtElapsed(elapsed)}
        </span>
      ) : activeCount > 0 ? (
        <span className="jobs-running-pill">Queued · {activeCount}</span>
      ) : null}
      <button
        ref={triggerRef}
        type="button"
        className={`jobs-trigger${activeCount > 0 ? ' has-active' : ''}`}
        aria-expanded={open}
        aria-haspopup="dialog"
        onClick={() => (open ? closePanel() : setOpen(true))}
      >
        <span className="jobs-dot" aria-hidden="true" />
        Jobs{activeCount > 0 ? ` · ${activeCount}` : ''}
      </button>
      {open ? (
        <div
          ref={panelRef}
          tabIndex={-1}
          className="jobs-panel"
          role="dialog"
          aria-label="Job queue"
        >
          <div className="jobs-panel-head">
            <h2 className="card-title">Job queue</h2>
            <button type="button" className="link-button" onClick={closePanel}>
              Close
            </button>
          </div>
          {jobs.length === 0 ? (
            <p className="muted">No jobs yet — pipeline steps run from their tabs.</p>
          ) : (
            <ul className="jobs-list">
              {jobs.map((entry) => (
                <JobRow
                  key={entry.id}
                  job={entry}
                  now={now}
                  selected={selected === entry.id}
                  onSelect={() => setSelectedId(entry.id)}
                />
              ))}
            </ul>
          )}
          {job ? (
            <div className="jobs-log-block">
              <div className="jobs-log-head">
                <h3 className="card-title">{job.label}</h3>
                <button type="button" className="link-button" onClick={copyLog}>
                  Copy log
                </button>
              </div>
              <LogView lines={job.log} maxHeight="14rem" />
              {job.status === 'failed' && job.error ? (
                <p className="jobs-error">{job.error}</p>
              ) : null}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}

function JobRow({
  job,
  now,
  selected,
  onSelect,
}: {
  job: JobSummary
  now: number
  selected: boolean
  onSelect: () => void
}) {
  const elapsed =
    job.status === 'running' && job.started_at !== null
      ? (now - job.started_at * 1000) / 1000
      : job.elapsed_s
  return (
    <li>
      <button
        type="button"
        className={`job-row${selected ? ' selected' : ''}`}
        onClick={onSelect}
      >
        <span className="job-row-label">{job.label}</span>
        <Badge variant={STATUS_VARIANT[job.status]}>{STATUS_LABEL[job.status]}</Badge>
        <span className="job-row-elapsed">{fmtElapsed(elapsed)}</span>
      </button>
    </li>
  )
}
