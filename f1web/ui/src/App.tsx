import { lazy, Suspense, useState, type KeyboardEvent } from 'react'
import { NextRace } from './components/next-race/NextRace'
import { RaceHistory } from './components/race-history/RaceHistory'
import { SeasonContext } from './components/season/SeasonContext'
import { Skeleton } from './components/ui/DataState'

// Chart-heavy views are lazy so Recharts isn't part of the initial bundle.
const Backtest = lazy(() =>
  import('./components/backtest/Backtest').then((m) => ({ default: m.Backtest })),
)
const Calibration = lazy(() =>
  import('./components/calibration/Calibration').then((m) => ({ default: m.Calibration })),
)
const Pipeline = lazy(() =>
  import('./components/pipeline/Pipeline').then((m) => ({ default: m.Pipeline })),
)
const Settings = lazy(() =>
  import('./components/settings/Settings').then((m) => ({ default: m.Settings })),
)

const TABS = [
  { id: 'next', label: 'Next Race', component: NextRace },
  { id: 'history', label: 'Race History', component: RaceHistory },
  { id: 'backtest', label: 'Backtest', component: Backtest },
  { id: 'calibration', label: 'Calibration', component: Calibration },
  { id: 'pipeline', label: 'Pipeline', component: Pipeline },
  { id: 'settings', label: 'Settings', component: Settings },
  { id: 'season', label: 'Season', component: SeasonContext },
] as const

type TabId = (typeof TABS)[number]['id']

export default function App() {
  const [tab, setTab] = useState<TabId>('next')
  const Active = TABS.find((entry) => entry.id === tab)?.component ?? TABS[0].component

  function moveFocus(event: KeyboardEvent, index: number) {
    if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return
    event.preventDefault()
    const next =
      event.key === 'ArrowRight'
        ? (index + 1) % TABS.length
        : (index - 1 + TABS.length) % TABS.length
    setTab(TABS[next].id)
    document.getElementById(`tab-${TABS[next].id}`)?.focus()
  }

  return (
    <div className="shell">
      <header className="site-header">
        <h1 className="brand">F1 Result Predictor</h1>
        <p className="brand-sub">Internal predictor dashboard</p>
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
            onClick={() => setTab(entry.id)}
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
        <Suspense fallback={<Skeleton rows={6} />}>
          <Active />
        </Suspense>
      </main>
    </div>
  )
}
