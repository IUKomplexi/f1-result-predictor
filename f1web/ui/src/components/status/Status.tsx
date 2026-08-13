import { useEffect, useState } from 'react'
import { getStatus, postJob, type Status as PipelineStatus } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { useJobs } from '../../hooks/useJob'
import { Badge } from '../ui/Badge'
import { ErrorState, Skeleton } from '../ui/DataState'
import './Status.css'

interface Step {
  id: string
  /** Job type to queue when the step is missing (see f1web/jobs.JOB_TYPES). */
  jobType: string
  /** Dashboard tab that hosts this step (navigation target). */
  tab: string
  tabLabel: string
  title: string
  desc: string
  ready: (status: PipelineStatus) => boolean
}

const STEPS: Step[] = [
  {
    id: 'fetch',
    jobType: 'fetch',
    tab: 'data',
    tabLabel: 'Data',
    title: 'Fetch raw data',
    desc: 'Downloads cached race results from the Jolpica API into data/raw. Everything after this works offline.',
    ready: (s) => s.data.has_raw_cache,
  },
  {
    id: 'train',
    jobType: 'train',
    tab: 'train',
    tabLabel: 'Train',
    title: 'Train the model',
    desc: 'Builds the featured dataset (data/features.parquet), trains the hurdle model checkpoint, and fits its probability calibrators.',
    ready: (s) => s.model.has_checkpoint && s.model.has_calibrators,
  },
  {
    id: 'backtest',
    jobType: 'backtest',
    tab: 'backtest',
    tabLabel: 'Backtest',
    title: 'Run a backtest',
    desc: 'Walk-forward validation of the model against grid / championship / zero baselines (reports/backtest.json).',
    ready: (s) => s.reports.has_backtest,
  },
]

/**
 * Pipeline onboarding: which of the three pipeline stages are ready, and a
 * one-click "run next" for the missing ones. Readiness comes from /api/status;
 * the list refreshes while any job is running so a finished job flips the
 * checklist without a manual reload.
 */
export function Status({ onNavigate }: { onNavigate?: (tabId: string) => void }) {
  const { state, retry } = useApi('status', () => getStatus())
  const { jobs } = useJobs()
  const [starting, setStarting] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const busy = jobs.some((j) => j.status === 'running' || j.status === 'queued')

  useEffect(() => {
    if (!busy) return
    const timer = setInterval(retry, 3000)
    return () => clearInterval(timer)
  }, [busy, retry])

  if (state.phase === 'loading') return <Skeleton rows={5} />
  if (state.phase === 'error') return <ErrorState message={state.message} onRetry={retry} />

  const status = state.data
  const readyCount = STEPS.filter((step) => step.ready(status)).length
  const complete = readyCount === STEPS.length

  const run = async (step: Step) => {
    setStarting(step.id)
    setError(null)
    try {
      await postJob(step.jobType)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setStarting(null)
    }
  }

  return (
    <section className="card">
      <div className="status-head">
        <div>
          <h2 className="card-title">Pipeline status</h2>
          <p className="meta-line">
            {readyCount} of {STEPS.length} steps ready — each step runs as a background
            job and builds on the previous one.
          </p>
        </div>
        <Badge variant={complete ? 'ready' : 'warn'}>
          {complete ? 'Ready to predict' : 'Setup incomplete'}
        </Badge>
      </div>

      <ol className="step-list">
        {STEPS.map((step, index) => {
          const ready = step.ready(status)
          return (
            <li key={step.id} className={`step${ready ? ' ready' : ''}`}>
              <span className="step-index">{index + 1}</span>
              <div className="step-body">
                <div className="step-title-row">
                  <h3 className="step-title">{step.title}</h3>
                  <Badge variant={ready ? 'ready' : 'missing'}>
                    {ready ? 'Ready' : 'Not ready'}
                  </Badge>
                </div>
                <p className="step-desc">{step.desc}</p>
              </div>
              <div className="step-actions">
                {ready ? (
                  <button
                    type="button"
                    className="button"
                    onClick={() => onNavigate?.(step.tab)}
                  >
                    Open {step.tabLabel}
                  </button>
                ) : (
                  <button
                    type="button"
                    className="button primary"
                    disabled={busy || starting === step.id}
                    onClick={() => run(step)}
                  >
                    {starting === step.id ? 'Queued…' : 'Run'}
                  </button>
                )}
              </div>
            </li>
          )
        })}
      </ol>

      {error ? <p className="save-status error">{error}</p> : null}
      <p className="muted status-note">
        The dataset itself is built automatically by Train (which also calibrates
        the model). Optional: precompute race history on Race History for instant
        past-race views.
      </p>
    </section>
  )
}
