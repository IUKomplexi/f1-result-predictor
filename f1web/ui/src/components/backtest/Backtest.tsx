import { useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { getBacktest, type Backtest, type BacktestMetricRow } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { fmtNumber } from '../../lib/format'
import { Badge } from '../ui/Badge'
import { ErrorState, Skeleton } from '../ui/DataState'
import { JobRunner } from '../ui/JobRunner'
import './Backtest.css'

const BASELINES = ['model', 'grid', 'championship', 'zero'] as const
type Baseline = (typeof BASELINES)[number]

const BASELINE_LABEL: Record<Baseline, string> = {
  model: 'Model',
  grid: 'Grid',
  championship: 'Championship',
  zero: 'Zero',
}

const BASELINE_COLOR: Record<Baseline, string> = {
  model: '#e10600',
  grid: '#4a7fd6',
  championship: '#d9a514',
  zero: '#6f6f7d',
}

const METRICS: { key: keyof BacktestMetricRow; label: string }[] = [
  { key: 'winner_hit', label: 'Winner hit' },
  { key: 'top3_overlap', label: 'Top-3 overlap' },
  { key: 'spearman', label: 'Spearman' },
  { key: 'mae', label: 'MAE' },
]

export function Backtest() {
  const [quantize, setQuantize] = useState(true)
  const [version, setVersion] = useState(0)
  const { state, retry } = useApi(`backtest-${version}`, () => getBacktest())
  return (
    <>
      <JobRunner
        type="backtest"
        runLabel="Run backtest"
        onDone={() => setVersion((v) => v + 1)}
        buildPayload={() => ({ quantize })}
        options={
          <div className="job-option">
            <label className="check-line" title="Round expected points to the nearest points-table value (matches the deployed predictor).">
              <input
                type="checkbox"
                checked={quantize}
                onChange={(e) => setQuantize(e.target.checked)}
              />
              Quantize points
            </label>
          </div>
        }
        renderResult={(job) => <BacktestRunResult job={job} />}
      />
      {state.phase === 'loading' ? (
        <Skeleton rows={8} />
      ) : state.phase === 'error' ? (
        <ErrorState message={state.message} onRetry={retry} />
      ) : (
        <BacktestView backtest={state.data} />
      )}
    </>
  )
}

function BacktestRunResult({ job }: { job: { result: Record<string, unknown> | null; log: string[] } }) {
  const overall = (job.result?.overall ?? {}) as Record<string, Record<string, number>>
  return (
    <div className="result-block">
      <h3 className="card-title">Backtest run</h3>
      {job.log.length > 0 && (
        <details className="job-log">
          <summary>Log</summary>
          {job.log.map((line, i) => (
            <pre key={i} className="log-line">{line}</pre>
          ))}
        </details>
      )}
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Baseline</th>
              <th scope="col" className="num">Winner hit</th>
              <th scope="col" className="num">Top3 overlap</th>
              <th scope="col" className="num">Spearman</th>
              <th scope="col" className="num">MAE</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(overall).map(([name, m]) => (
              <tr key={name}>
                <td>{name}</td>
                <td className="num">{m.winner_hit?.toFixed(3)}</td>
                <td className="num">{m.top3_overlap?.toFixed(3)}</td>
                <td className="num">{m.spearman?.toFixed(3)}</td>
                <td className="num">{m.mae?.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function BacktestView({ backtest }: { backtest: Backtest }) {
  const seasons = allSeasons(backtest.by_season)

  return (
    <>
      <section className="card">
        <h2 className="card-title">Model vs baselines (walk-forward, mean)</h2>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Baseline</th>
                {METRICS.map((metric) => (
                  <th key={metric.key} scope="col" className="num">
                    {metric.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {BASELINES.map((baseline) => (
                <tr key={baseline} className={baseline === 'model' ? 'row-model' : undefined}>
                  <td>
                    {baseline === 'model' ? (
                      <Badge variant="info">{BASELINE_LABEL[baseline]}</Badge>
                    ) : (
                      BASELINE_LABEL[baseline]
                    )}
                  </td>
                  {METRICS.map((metric) => (
                    <td key={metric.key} className="num">
                      {fmtNumber(backtest.overall[baseline]?.[metric.key], 3)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <h2 className="card-title">Per-season trends</h2>
        <div className="chart-grid">
          {METRICS.map((metric) => (
            <figure key={metric.key} className="chart-figure">
              <figcaption>{metric.label}</figcaption>
              <MetricChart
                metric={metric.key}
                seasons={seasons}
                bySeason={backtest.by_season}
              />
            </figure>
          ))}
        </div>
      </section>
    </>
  )
}

function allSeasons(bySeason: Record<string, Record<string, BacktestMetricRow>>): number[] {
  const set = new Set<number>()
  for (const table of Object.values(bySeason)) {
    for (const season of Object.keys(table)) set.add(Number(season))
  }
  return [...set].sort((a, b) => a - b)
}

function MetricChart({
  metric,
  seasons,
  bySeason,
}: {
  metric: keyof BacktestMetricRow
  seasons: number[]
  bySeason: Record<string, Record<string, BacktestMetricRow>>
}) {
  const data = seasons.map((season) => {
    const row: Record<string, number | string | undefined> = {
      season: String(season),
    }
    for (const baseline of BASELINES) {
      row[baseline] = bySeason[baseline]?.[String(season)]?.[metric]
    }
    return row
  })

  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
          <CartesianGrid stroke="#2e2e3a" strokeDasharray="3 3" />
          <XAxis
            dataKey="season"
            tick={{ fill: '#a8a8b5', fontSize: 11 }}
            stroke="#2e2e3a"
          />
          <YAxis
            tick={{ fill: '#a8a8b5', fontSize: 11 }}
            stroke="#2e2e3a"
            domain={['auto', 'auto']}
          />
          <Tooltip
            contentStyle={{
              background: '#23232e',
              border: '1px solid #2e2e3a',
              borderRadius: 8,
              color: '#f2f2f5',
            }}
          />
          <Legend wrapperStyle={{ fontSize: 12, color: '#a8a8b5' }} />
          {BASELINES.map((baseline) => (
            <Line
              key={baseline}
              type="linear"
              dataKey={baseline}
              name={BASELINE_LABEL[baseline]}
              stroke={BASELINE_COLOR[baseline]}
              strokeWidth={baseline === 'model' ? 2.5 : 1.5}
              dot={false}
              connectNulls={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
