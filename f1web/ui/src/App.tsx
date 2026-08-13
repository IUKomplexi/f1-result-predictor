import {
  lazy,
  Suspense,
  useEffect,
  useRef,
  useState,
  type ComponentType,
  type KeyboardEvent,
} from 'react'
import { getStatus } from './api/client'
import { Race } from './components/race/Race'
import { RaceHistory } from './components/race-history/RaceHistory'
import { OverridePrediction } from './components/override/OverridePrediction'
import { SeasonContext } from './components/season/SeasonContext'
import { Status } from './components/status/Status'
import { ErrorBoundary } from './components/ui/ErrorBoundary'
import { Skeleton } from './components/ui/DataState'
import { JobsWidget } from './components/ui/JobsWidget'

// Chart-heavy views are lazy so Recharts isn't part of the initial bundle.
const Backtest = lazy(() =>
  import('./components/backtest/Backtest').then((m) => ({ default: m.Backtest })),
)
const Calibration = lazy(() =>
  import('./components/calibration/Calibration').then((m) => ({ default: m.Calibration })),
)
const Data = lazy(() =>
  import('./components/data/Data').then((m) => ({ default: m.Data })),
)
const Train = lazy(() =>
  import('./components/train/Train').then((m) => ({ default: m.Train })),
)
const Search = lazy(() =>
  import('./components/search/Search').then((m) => ({ default: m.Search })),
)
const Settings = lazy(() =>
  import('./components/settings/Settings').then((m) => ({ default: m.Settings })),
)

type TabId =
  | 'status'
  | 'race'
  | 'history'
  | 'data'
  | 'train'
  | 'search'
  | 'backtest'
  | 'calibration'
  | 'specific'
  | 'settings'
  | 'season'

interface TabEntry {
  id: TabId
  label: string
  component: ComponentType<{ onNavigate?: (tabId: string) => void }>
}

const TABS: TabEntry[] = [
  { id: 'status', label: 'Status', component: Status },
  { id: 'race', label: 'Race', component: Race },
  { id: 'history', label: 'Race History', component: RaceHistory },
  { id: 'data', label: 'Data', component: Data },
  { id: 'train', label: 'Train', component: Train },
  { id: 'search', label: 'Search', component: Search },
  { id: 'backtest', label: 'Backtest', component: Backtest },
  { id: 'calibration', label: 'Calibration', component: Calibration },
  { id: 'specific', label: 'Specific Race', component: OverridePrediction },
  { id: 'settings', label: 'Settings', component: Settings },
  { id: 'season', label: 'Season', component: SeasonContext },
]

export default function App() {
  // Race is the default once the pipeline is ready; a first-run setup that
  // has missing artifacts lands on Status instead (see the effect below).
  const [tab, setTab] = useState<TabId>('race')
  const userChose = useRef(false)
  const Active = TABS.find((entry) => entry.id === tab)?.component ?? TABS[0].component

  useEffect(() => {
    let cancelled = false
    getStatus()
      .then((status) => {
        if (cancelled || userChose.current) return
        const ready =
          status.data.has_raw_cache &&
          status.model.has_checkpoint &&
          status.model.has_calibrators &&
          status.reports.has_backtest
        if (!ready) setTab('status')
      })
      .catch(() => {
        // Status unavailable (e.g. backend not built): keep the Race default.
      })
    return () => {
      cancelled = true
    }
  }, [])

  function selectTab(id: TabId) {
    userChose.current = true
    setTab(id)
  }

  function moveFocus(event: KeyboardEvent, index: number) {
    if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return
    event.preventDefault()
    const next =
      event.key === 'ArrowRight'
        ? (index + 1) % TABS.length
        : (index - 1 + TABS.length) % TABS.length
    selectTab(TABS[next].id)
    document.getElementById(`tab-${TABS[next].id}`)?.focus()
  }

  return (
    <div className="shell">
      <header className="site-header">
        <h1 className="brand">F1 Result Predictor</h1>
        <p className="brand-sub">Internal predictor dashboard</p>
        <JobsWidget />
      </header>
      <nav className="tabs" role="tablist" aria-label="Dashboard sections">
        {TABS.map((entry, index) => (
          <button
            key={entry.id}
            type="button"
            role="tab"
            id={`tab-${entry.id}`}
            aria-selected={tab === entry.id}
            aria-controls="dashboard-panel"
            className={`tab${tab === entry.id ? ' active' : ''}`}
            onClick={() => selectTab(entry.id)}
            onKeyDown={(event) => moveFocus(event, index)}
          >
            {entry.label}
          </button>
        ))}
      </nav>
      <main
        id="dashboard-panel"
        role="tabpanel"
        aria-labelledby={`tab-${tab}`}
        className="panel"
      >
        <ErrorBoundary>
          <Suspense fallback={<Skeleton rows={6} />}>
            <Active onNavigate={(id) => selectTab(id as TabId)} />
          </Suspense>
        </ErrorBoundary>
      </main>
    </div>
  )
}
