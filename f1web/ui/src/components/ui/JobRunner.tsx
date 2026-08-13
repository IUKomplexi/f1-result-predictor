import { useEffect, useRef, useState, type ReactNode } from 'react'
import { postJob, type Job } from '../../api/client'
import { useJob, useJobs } from '../../hooks/useJob'
import { LogView } from './LogView'
import './JobRunner.css'

interface JobRunnerProps {
  /** The job type (fetch / train / calibrate / backtest / search). */
  type: string
  /** Run-button label, e.g. "Fetch data". */
  runLabel?: string
  /** Optional controls rendered alongside the Run button (inputs, toggles). */
  options?: ReactNode
  /** Build the job payload from the current option state. */
  buildPayload: () => Record<string, unknown>
  /** Render the JSON-safe result of a finished job. */
  renderResult: (job: Job) => ReactNode
  /** Called once when a run reaches done/failed (e.g. to refresh a report). */
  onDone?: () => void
}

/**
 * Shared inline pipeline-step control: a Run button (with optional option
 * controls), a live log while a job is running, and the result of either the
 * just-run job or the most recent finished job of this type. All five pipeline
 * tabs use this so they look and behave consistently.
 */
export function JobRunner({
  type,
  runLabel = 'Run',
  options,
  buildPayload,
  renderResult,
  onDone,
}: JobRunnerProps) {
  const { jobs } = useJobs()
  const [currentId, setCurrentId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const latestDone = jobs.find((j) => j.type === type && j.status === 'done')
  const pollId = currentId ?? latestDone?.id ?? null
  const job = useJob(pollId)
  const anyBusy = jobs.some((j) => j.status === 'running' || j.status === 'queued')
  const busyJob = jobs.find((j) => j.status === 'running' || j.status === 'queued') ?? null
  const running = job?.status === 'running' || job?.status === 'queued'

  // Notify onDone once when a tracked run reaches a terminal state.
  const notified = useRef<string | null>(null)
  useEffect(() => {
    const id = job?.id ?? null
    const status = job?.status
    if (status === 'done' || status === 'failed') {
      if (notified.current !== id) {
        notified.current = id
        onDone?.()
      }
    }
  }, [job?.status, job?.id, onDone])

  const run = async () => {
    setSubmitting(true)
    setError(null)
    try {
      const { id } = await postJob(type, buildPayload())
      setCurrentId(id)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section className="card">
      <div className="job-options">
        {options}
        <button type="button" className="button" onClick={run} disabled={submitting || anyBusy}>
          {submitting ? 'Queued…' : runLabel}
        </button>
      </div>
      {error ? <p className="save-status error">{error}</p> : null}
      {anyBusy && !running && busyJob ? (
        <p className="jobs-busy-note">
          {busyJob.label} is {busyJob.status === 'running' ? 'running' : 'queued'} — Run is
          paused until the queue clears (see the Jobs button up top).
        </p>
      ) : null}
      {job && running ? (
        <div className="job-log">
          <h3 className="card-title">
            {job.label} — {job.status === 'queued' ? 'queued' : 'running'}
          </h3>
          <LogView lines={job.log} maxHeight="16rem" />
        </div>
      ) : null}
      {job && job.status === 'failed' ? (
        <p className="save-status error">Failed: {job.error ?? 'unknown error'}</p>
      ) : null}
      {job && job.status === 'done'
        ? renderResult(job)
        : pollId === null
          ? <p className="muted">No run yet.</p>
          : null}
    </section>
  )
}
