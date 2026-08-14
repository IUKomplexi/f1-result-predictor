import { fmtNumber } from '../../lib/format'
import type { Backtest, BacktestMetricRow } from '../../api/client'
import { Chart, type ChartDatum, type ChartSeries } from '../ui/Chart'
import {
  BASELINES,
  BASELINE_COLOR,
  BASELINE_LABEL,
  METRICS,
  MODEL_COLORS,
  allSeasons,
  deltaVs,
} from './lib'
import { MetricCell } from './MetricCell'

/** Per-metric charts (2×2) plus absolute + delta tables for the compared models. */
export function ModelComparison({ backtest, reference }: { backtest: Backtest; reference: string | null }) {
  const byModel = backtest.models ?? {}
  const names = Object.keys(byModel).sort()
  const refName = reference && names.includes(reference) ? reference : names[0]
  return (
    <>
      <section className="card">
        <h2 className="card-title">Model comparison</h2>
        <div className="chart-grid-2">
          {METRICS.map((metric) => (
            <figure key={metric.key} className="chart-figure">
              <figcaption>{metric.label} — per model</figcaption>
              <CompareChart
                metric={metric.key}
                names={names}
                byModel={byModel}
                bySeason={backtest.by_season}
              />
            </figure>
          ))}
        </div>
      </section>

      <section className="card">
        <h2 className="card-title">Model scores (mean over the backtest range)</h2>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Model</th>
                {METRICS.map((metric) => (
                  <th key={metric.key} scope="col" className="num">
                    {metric.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {names.map((name) => (
                <tr key={name} className={name === refName ? 'row-model' : undefined}>
                  <td>
                    {name}
                    {name === refName ? <span className="muted"> (reference)</span> : null}
                  </td>
                  {METRICS.map((metric) => (
                    <MetricCell
                      key={metric.key}
                      metric={metric.key}
                      values={names.map(
                        (n) => byModel[n]?.overall.model?.[metric.key] ?? NaN,
                      )}
                      value={byModel[name]?.overall.model?.[metric.key]}
                    />
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="cell-tone-legend">
          Green = best model for that metric, red = worst, orange = in between,
          white = tied. MAE is lower-better; the other metrics are higher-better.
        </p>
        <CompareDeltaTable byModel={byModel} reference={refName} />
      </section>
    </>
  )
}

function CompareChart({
  metric,
  names,
  byModel,
  bySeason,
}: {
  metric: keyof BacktestMetricRow
  names: string[]
  byModel: NonNullable<Backtest['models']>
  bySeason: Record<string, Record<string, BacktestMetricRow>>
}) {
  const seasons = allSeasons(bySeason)
  const data: ChartDatum[] = seasons.map((season) => {
    const row: ChartDatum = { season: String(season) }
    for (const name of names) {
      row[name] = byModel[name]?.by_season.model?.[String(season)]?.[metric]
    }
    for (const baseline of BASELINES) {
      row[`b_${baseline}`] = bySeason[baseline]?.[String(season)]?.[metric]
    }
    return row
  })
  const series: ChartSeries[] = [
    ...names.map((name, i) => ({
      key: name,
      name,
      color: MODEL_COLORS[i % MODEL_COLORS.length],
      strokeWidth: 2,
    })),
    ...BASELINES.map((baseline) => ({
      key: `b_${baseline}`,
      name: BASELINE_LABEL[baseline],
      color: BASELINE_COLOR[baseline],
      strokeWidth: 1.5,
    })),
  ]
  return (
    <Chart data={data} xKey="season" series={series} valueFormat={(v) => fmtNumber(v, 3)} />
  )
}

/** Overall deltas vs the reference model (positive = better for every metric). */
function CompareDeltaTable({
  byModel,
  reference,
}: {
  byModel: NonNullable<Backtest['models']>
  reference: string
}) {
  const names = Object.keys(byModel).sort()
  const refRow = byModel[reference]?.overall.model
  return (
    <>
      <h3 className="card-title">
        Deltas vs {reference}
      </h3>
      <p className="cell-tone-legend">
        Each cell is that model's value minus {reference}'s. Positive = better
        for every metric (MAE is inverted).
      </p>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Model</th>
              {METRICS.map((metric) => (
                <th key={metric.key} scope="col" className="num">
                  Δ {metric.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {names.map((name) => {
              const isRef = name === reference
              return (
                <tr key={name} className={isRef ? 'row-model' : undefined}>
                  <td>{name}{isRef ? ' (reference)' : ''}</td>
                  {METRICS.map((metric) => {
                    const delta = deltaVs(byModel[name]?.overall.model, refRow, metric.key)
                    if (delta === null) {
                      return (
                        <td key={metric.key} className="num muted">–</td>
                      )
                    }
                    return (
                      <td key={metric.key} className="num">
                        <span className={delta >= 0 ? 'delta-pos' : 'delta-neg'}>
                          {fmtNumber(delta, 3, true)}
                        </span>
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </>
  )
}
