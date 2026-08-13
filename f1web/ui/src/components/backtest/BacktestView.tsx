import { fmtNumber } from '../../lib/format'
import type { Backtest, BacktestMetricRow } from '../../api/client'
import { Badge } from '../ui/Badge'
import { Chart, type ChartDatum, type ChartSeries } from '../ui/Chart'
import {
  BASELINES,
  BASELINE_COLOR,
  BASELINE_LABEL,
  METRICS,
  allSeasons,
  deltaVs,
} from './lib'
import { ModelComparison } from './ModelComparison'

/** The persisted backtest report: mean table, model comparison, trend + edge charts. */
export function BacktestView({ backtest, reference }: { backtest: Backtest; reference: string | null }) {
  const seasons = allSeasons(backtest.by_season)

  return (
    <>
      <section className="card">
        <h2 className="card-title">Model vs baselines (mean)</h2>
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

      {backtest.models && Object.keys(backtest.models).length > 1 ? (
        <ModelComparison backtest={backtest} reference={reference} />
      ) : null}

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

      <section className="card">
        <h2 className="card-title">Model edge vs baselines (per season)</h2>
        <p className="context-note">
          Positive means the model beat the baseline that season (for MAE,
          lower is better, so the edge is reversed). See how much better/worse
          the model is than the grid / championship across seasons.
        </p>
        <div className="chart-grid">
          {METRICS.map((metric) => (
            <figure key={metric.key} className="chart-figure">
              <figcaption>vs grid / championship — {metric.label}</figcaption>
              <MetricEdgeChart
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

function MetricChart({
  metric,
  seasons,
  bySeason,
}: {
  metric: keyof BacktestMetricRow
  seasons: number[]
  bySeason: Record<string, Record<string, BacktestMetricRow>>
}) {
  const data: ChartDatum[] = seasons.map((season) => {
    const row: ChartDatum = {
      season: String(season),
    }
    for (const baseline of BASELINES) {
      row[baseline] = bySeason[baseline]?.[String(season)]?.[metric]
    }
    return row
  })
  const series: ChartSeries[] = BASELINES.map((baseline) => ({
    key: baseline,
    name: BASELINE_LABEL[baseline],
    color: BASELINE_COLOR[baseline],
    strokeWidth: baseline === 'model' ? 2.5 : 1.5,
  }))
  return (
    <Chart data={data} xKey="season" series={series} valueFormat={(v) => fmtNumber(v, 3)} />
  )
}

function MetricEdgeChart({
  metric,
  seasons,
  bySeason,
}: {
  metric: keyof BacktestMetricRow
  seasons: number[]
  bySeason: Record<string, Record<string, BacktestMetricRow>>
}) {
  const data: ChartDatum[] = seasons.map((season) => {
    const s = String(season)
    return {
      season: s,
      vsGrid: deltaVs(bySeason.model?.[s], bySeason.grid?.[s], metric),
      vsChamp: deltaVs(bySeason.model?.[s], bySeason.championship?.[s], metric),
    }
  })
  const series: ChartSeries[] = [
    { key: 'vsGrid', name: 'vs grid', color: '#4a7fd6', strokeWidth: 2 },
    { key: 'vsChamp', name: 'vs championship', color: '#d9a514', strokeWidth: 2 },
  ]
  return (
    <Chart
      data={data}
      xKey="season"
      series={series}
      referenceLine={{ y: 0, label: 'even' }}
      valueFormat={(v) => fmtNumber(v, 3)}
    />
  )
}
