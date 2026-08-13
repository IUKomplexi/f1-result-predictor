import type { Status as PipelineStatus } from '../../api/client'

export interface Step {
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

/**
 * The pipeline onboarding checklist, separate from the presentation logic so
 * the readiness configuration is easy to scan and edit without touching the
 * rendering.
 */
export const STEPS: Step[] = [
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
