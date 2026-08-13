import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ComponentType,
  type KeyboardEvent,
} from 'react'
import { getStatus } from './api/client'
import { Race } from './components/race/Race'
import { RaceHistory } from './components/race-history/RaceHistory'
import { Status } from './components/status/Status'
import { ErrorBoundary } from './components/ui/ErrorBoundary'
import { Skeleton } from './components/ui/DataState'
import { JobsWidget } from './components/ui/JobsWidget'
import { Badge } from './components/ui/Badge'

// Chart-heavy views are lazy so Recharts isn't part of the initial bundle.
const Backtest = lazy(() =>
  import('./components/backtest/Backtest').then((m) => ({ default: m.Backtest })),
)
const Data = lazy(() =>
  import('./components/data/Data').then((m) => ({ default: m.Data })),
)
const Train = lazy(() =>
  import('./components/train/Train').then((m) => ({ default: m.Train })),
)
const Settings = lazy(() =>
  import('./components/settings/Settings').then((m) => ({ default: m.Settings })),
)

type TabId = 'status' | 'race' | 'history' | 'data' | 'train' | 'backtest' | 'settings'

/** Cross-tab navigation payload (e.g. Race History → a specific race). */
export interface NavState {
  season?: number
  round?: number
}

export interface TabProps {
  onNavigate?: (tabId: string, state?: NavState) => void
  navState?: NavState | null
  /** A tab can veto navigation (e.g. Settings with unsaved edits). The guard
   *  returns false to block; the component retries via onNavigate itself. */
  setNavigateGuard?: (
    guard: ((tabId: string, state?: NavState) => boolean) | null,
  ) => void
}

interface TabEntry {
  id: TabId
  label: string
  component: ComponentType<TabProps>
}

const TABS: TabEntry[] = [
  { id: 'status', label: 'Status', component: Status },
  { id: 'race', label: 'Race', component: Race },
  { id: 'history', label: 'Race History', component: RaceHistory },
  { id: 'data', label: 'Data', component: Data },
  { id: 'train', label: 'Train', component: Train },
  { id: 'backtest', label: 'Backtest', component: Backtest },
  { id: 'settings', label: 'Settings', component: Settings },
]

const TAB_IDS = new Set<string>(TABS.map((entry) => entry.id))

/** The tab encoded in the URL hash (`#/backtest`), or null when absent/unknown. */
function tabFromHash(): TabId | null {
  const hash = window.location.hash.replace(/^#\/?/, '')
  return TAB_IDS.has(hash) ? (hash as TabId) : null
}

export default function App() {
  // Race is the default once the pipeline is ready; a first-run setup that
  // has missing artifacts lands on Status instead (see the effect below).
  // A deep link (e.g. #/backtest) wins: it counts as a user choice.
  const initialHash = useRef(tabFromHash())
  const [tab, setTab] = useState<TabId>(() => initialHash.current ?? 'race')
  const [navState, setNavState] = useState<NavState | null>(null)
  const [pipelineReady, setPipelineReady] = useState<boolean | null>(null)
  const userChose = useRef(initialHash.current !== null)
  const navigateGuard = useRef<((tabId: string, state?: NavState) => boolean) | null>(null)
  const Active = TABS.find((entry) => entry.id === tab)?.component ?? TABS[0].component

  useEffect(() => {
    let cancelled = false
    getStatus()
      .then((status) => {
        if (cancelled) return
        const ready =
          status.data.has_raw_cache &&
          status.model.has_checkpoint &&
          status.model.has_calibrators &&
          status.reports.has_backtest
        setPipelineReady(ready)
        if (!userChose.current && !ready) selectTab('status')
      })
      .catch(() => {
        // Status unavailable (e.g. backend not built): keep the Race default.
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Keep the tab in sync with the URL hash: browser back/forward and
  // middle-click open (open in new tab) both work via hashchange.
  useEffect(() => {
    function onHashChange() {
      const fromHash = tabFromHash()
      if (fromHash !== null) selectTab(fromHash)
    }
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  function selectTab(id: TabId, state?: NavState) {
    if (navigateGuard.current && !navigateGuard.current(id, state)) return
    userChose.current = true
    setTab(id)
    setNavState(state ?? null)
    if (tabFromHash() !== id) {
      // Push a history entry; the hashchange listener above picks it up
      // (and no-ops, since state already matches).
      window.location.hash = `/${id}`
    }
  }

  const setNavigateGuard = useCallback(
    (guard: ((tabId: string, state?: NavState) => boolean) | null) => {
      navigateGuard.current = guard
    },
    [],
  )

  function moveFocus(event: KeyboardEvent, index: number) {
    let next: number | null = null
    if (event.key === 'ArrowRight') next = (index + 1) % TABS.length
    else if (event.key === 'ArrowLeft') next = (index - 1 + TABS.length) % TABS.length
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = TABS.length - 1
    if (next === null) return
    event.preventDefault()
    selectTab(TABS[next].id)
    document.getElementById(`tab-${TABS[next].id}`)?.focus()
  }

  return (
    <div className="shell">
      <header className="site-header">
        <h1 className="brand">F1 Result Predictor</h1>
        <p className="brand-sub">Internal predictor dashboard</p>
        {pipelineReady === false ? (
          <span className="header-alert" role="status">
            <Badge variant="warn">Pipeline not ready</Badge>
            <button
              type="button"
              className="link-button"
              onClick={() => selectTab('status')}
            >
              Open Status
            </button>
          </span>
        ) : null}
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
            tabIndex={tab === entry.id ? 0 : -1}
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
            <Active onNavigate={selectTab} navState={navState} setNavigateGuard={setNavigateGuard} />
          </Suspense>
        </ErrorBoundary>
      </main>
    </div>
  )
}
