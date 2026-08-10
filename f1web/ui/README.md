# f1-dashboard (React SPA)

Browser dashboard for the F1 result predictor. Consumes the internal JSON API
(`/api/*`) served by the FastAPI backend in `f1web/app.py`.

## Development

With the FastAPI backend running on port 8080:

```sh
npm install
npm run dev        # http://localhost:5173 — /api is proxied to :8080
```

## Production build

```sh
npm run build      # writes f1web/ui/dist
```

The FastAPI backend serves the built SPA at `/` and `/dashboard` (see
`f1web/app.py`), so after `npm run build` just start the backend:

```sh
f1-web --port 8080   # http://127.0.0.1:8080/
```

## Layout

- `src/api/client.ts` — typed client for the predictor API (all endpoints).
- `src/theme/tokens.css` — F1-themed design tokens (colors, spacing, type).
- `src/App.tsx` — app shell (Next Race / Race History / Backtest / Calibration /
  Season tabs).
- Pinned to stable releases (React 18, Vite 5, TypeScript 5.5, Recharts 2).
