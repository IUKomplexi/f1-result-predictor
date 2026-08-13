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

/** Per-metric charts with one series per compared model plus the baselines. */
export function ModelComparison({ backtest, reference }: { backtest: Backtest; reference: string | null }) {
  const byModel = backtest.models ?? {}
  const names = Object.keys(byModel).sort()
  const refName = reference && names.includes(reference) ? reference : names[0]
  return (
    <section className="card">
      <h2 className="card-title">Model comparison</h2>
      <p className="context-note">
        All models are scored on the same seasons. Baselines are identical, so
        only the model lines differ.
      </p>
      <div className="chart-grid">
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
      <CompareDeltaTable byModel={byModel} reference={refName} />
    </section>
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

/** Overall deltas vs the reference model (positive = better). */
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
      <h3 className="card-title">Deltas vs {reference} (positive = better)</h3>
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
