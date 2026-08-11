import { useEffect, useState } from 'react'
import { getJob, getJobs, type Job, type JobSummary } from '../api/client'

const POLL_MS = 1500

/**
 * Poll the job list while any job is still queued/running, then settle.
 * Re-fetches immediately whenever a new job id is added via `refresh`.
 */
export function useJobs(): { jobs: JobSummary[]; refresh: () => void } {
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [tick, setTick] = useState(0)

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setInterval> | undefined

    const load = async () => {
      try {
        const data = await getJobs()
        if (cancelled) return
        setJobs(data.jobs)
        const busy = data.jobs.some((j) => j.status === 'queued' || j.status === 'running')
        if (busy && timer === undefined) {
          timer = setInterval(() => {
            load()
          }, POLL_MS)
        } else if (!busy && timer !== undefined) {
          clearInterval(timer)
          timer = undefined
        }
      } catch {
        // Transient failure: keep polling on the next tick.
      }
    }
    load()
    return () => {
      cancelled = true
      if (timer !== undefined) clearInterval(timer)
    }
  }, [tick])

  return { jobs, refresh: () => setTick((n) => n + 1) }
}

/**
 * Poll a single job until it reaches a terminal state; returns live data.
 * Re-subscribes when `id` changes.
 */
export function useJob(id: string | null): Job | null {
  const [job, setJob] = useState<Job | null>(null)

  useEffect(() => {
    if (!id) {
      setJob(null)
      return
    }
    let cancelled = false
    let timer: ReturnType<typeof setInterval> | undefined

    const load = async () => {
      try {
        const data = await getJob(id)
        if (cancelled) return
        setJob(data)
        const done = data.status === 'done' || data.status === 'failed'
        if (!done && timer === undefined) {
          timer = setInterval(() => load(), POLL_MS)
        } else if (done && timer !== undefined) {
          clearInterval(timer)
          timer = undefined
        }
      } catch {
        // keep polling
      }
    }
    load()
    return () => {
      cancelled = true
      if (timer !== undefined) clearInterval(timer)
    }
  }, [id])

  return job
}
