# f1-dashboard (Preact SPA)

Browser dashboard for the F1 result predictor. Consumes the internal JSON API
(`/api/*`) served by the FastAPI backend in `f1web/app.py`.

## Development

With the FastAPI backend running on port 8080:

```sh
npm install
npm run dev        # http://localhost:5173 — /api is proxied to :8080
npm test           # vitest component + lib tests (offline, jsdom)
```

## Production build

```sh
npm run build      # writes f1web/ui/dist
```

The FastAPI backend serves the built SPA at `/` and `/dashboard` (see
`f1web/app.py`), so after `npm run build` just start the backend:

```sh
f1 web --port 8080   # http://127.0.0.1:8080/
```

## Navigating the code (start here)

The app is a single page with 7 tabs. The rule of thumb:

- **`src/pages/<tab>/`** — one folder per tab. The tab's **entry component is
  the file named after the tab** (`Race.tsx`, `Backtest.tsx`, …) — start there
  to read a tab.
- **`src/components/`** — reusable building blocks shared across tabs, split by
  role: `ui/` (visual primitives), `jobs/` (job execution widgets),
  `controls/` (form controls).
- **`src/hooks/`**, **`src/lib/`**, **`src/api/`** — shared React hooks, shared
  pure functions, and the typed API client.
- **`src/App.tsx`** — the shell: renders the tab bar, switches tabs, and
  keeps the active tab in the URL hash (`#/race`, `#/backtest`, …).
- **`src/types.ts`** — the `TabProps`/`NavState` contract every tab receives
  (how tabs navigate to each other).

### Directory map

```
src/
  main.tsx            # entry point — mounts <App/> (no app logic here)
  App.tsx             # app shell: tab bar + hash routing + lazy tab loading
  types.ts            # TabProps / NavState — the cross-tab contract
  index.css           # global styles
  api/
    client.ts         # typed fetch client for every /api endpoint
  pages/              # one folder per dashboard tab
    status/           #   pipeline health + first-run setup steps
    race/             #   one race's prediction; grid + model overrides
    race-history/     #   a season's predictions vs actuals, round by round
    data/             #   raw-data fetch + dataset build jobs
    train/            #   train (+ calibrate) job
    backtest/         #   walk-forward backtest + model comparison
    calibration/      #   calibration report (Brier + reliability curves)
    settings/         #   config.toml editor
  components/         # shared building blocks (reused across pages)
    ui/               #   visual primitives: Badge, Chart, DataState, LogView, …
    jobs/             #   job execution widgets: JobRunner, JobsWidget
    controls/         #   form controls: SeasonRange, FeatureToggles, …
  hooks/              # shared React hooks: useApi, useJob
  lib/                # shared pure functions (no React)
    format.ts         #   display formatting (points, dates, elapsed)
    models.ts         #   model-selection helpers (deployed name, choices)
    analysis.ts       #   prediction → Race History summary row
    debounce.ts       #   debounced callbacks
  theme/
    tokens.css        # F1-themed design tokens (colors, spacing, type)
```

### Tab → entry file

| Tab label        | Entry file                     | What it does                                        |
| ---------------- | ------------------------------ | --------------------------------------------------- |
| Status           | `pages/status/Status.tsx`      | Pipeline health, first-run setup checklist          |
| Race             | `pages/race/Race.tsx`          | Predict one race; edit the qualifying grid          |
| Race History     | `pages/race-history/RaceHistory.tsx` | Season overview of predictions vs actuals      |
| Data             | `pages/data/Data.tsx`          | Fetch raw data, build the dataset                   |
| Train            | `pages/train/Train.tsx`        | Train + calibrate the model                         |
| Backtest         | `pages/backtest/Backtest.tsx`  | Score models walk-forward vs baselines              |
| Settings         | `pages/settings/Settings.tsx`  | Edit `config.toml` from the browser                 |

### How tabs talk to each other

Tabs are independent components receiving the `TabProps` contract from
`src/types.ts`. To jump from one tab to another (e.g. Race History → a
specific race), a tab calls `onNavigate('race', { season, round })`. Any tab
can veto navigation with `setNavigateGuard` (Settings uses this to block
leaving with unsaved edits). The active tab is always encoded in the URL hash,
so refresh and back/forward work.

### Conventions

- Tests live next to their component as `*.test.{ts,tsx}` (offline, jsdom —
  no network).
- Pages and shared components keep their own CSS file next to them, imported
  via `import './X.css'`.
- Shared pure logic goes in `src/lib/`; per-tab logic goes in the tab folder
  as `lib.ts` (e.g. `pages/race/lib.ts`).
- Pinned to stable releases (Preact 10, Vite 5, TypeScript 5.5, Vitest 2).
