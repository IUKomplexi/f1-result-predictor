import { fmtNumber } from '../../lib/format'
import type { Backtest, BacktestMetricRow } from '../../api/client'
import { Badge } from '../ui/Badge'
import { Chart, type ChartDatum, type ChartSeries } from '../ui/Chart'
import {
  BASELINES,
  BASELINE_HELP,
  BASELINE_LABEL,
  METRICS,
  allSeasons,
  deltaVs,
} from './lib'
import { MetricCell } from './MetricCell'
import { ModelComparison } from './ModelComparison'

/** Plain-language meaning of each metric (shown under the mean table). */
const METRIC_HELP: Record<string, string> = {
  winner_hit: 'how often the predicted winner actually won',
  top3_overlap: 'share of the actual podium the model named',
  top10_overlap: 'share of the actual top 10 the model named',
  spearman: 'rank correlation of predicted vs actual points order',
  mae: 'mean absolute error of predicted points',
}

function metricLabel(metric: keyof BacktestMetricRow): string {
  return METRICS.find((m) => m.key === metric)?.label ?? String(metric)
}

/** The persisted backtest report: mean table, model comparison, edge charts. */
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
              {BASELINES.map((baseline) => {
                return (
                  <tr key={baseline} className={baseline === 'model' ? 'row-model' : undefined}>
                    <td>
                      {baseline === 'model' ? (
                        <Badge variant="info">{BASELINE_LABEL[baseline]}</Badge>
                      ) : (
                        BASELINE_LABEL[baseline]
                      )}
                    </td>
                    {METRICS.map((metric) => (
                      <MetricCell
                        key={metric.key}
                        metric={metric.key}
                        values={BASELINES.map(
                          (b) => backtest.overall[b]?.[metric.key] ?? NaN,
                        )}
                        value={backtest.overall[baseline]?.[metric.key]}
                      />
                    ))}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <p className="cell-tone-legend">
          Green = best in the column, red = worst, orange = in between, white =
          tied. MAE is lower-better; the other metrics are higher-better.
        </p>
        <dl className="metric-glossary">
          {METRICS.map((metric) => (
            <div key={metric.key} className="glossary-item">
              <dt>{metric.label}</dt>
              <dd>{METRIC_HELP[metric.key]}</dd>
            </div>
          ))}
          {BASELINES.filter((b) => b !== 'model').map((baseline) => (
            <div key={baseline} className="glossary-item">
              <dt>{BASELINE_LABEL[baseline]}</dt>
              <dd>{BASELINE_HELP[baseline]}</dd>
            </div>
          ))}
        </dl>
      </section>

      {backtest.models && Object.keys(backtest.models).length > 1 ? (
        <ModelComparison backtest={backtest} reference={reference} />
      ) : null}

      <section className="card">
        <h2 className="card-title">Model edge vs baselines (per season)</h2>
        <p className="context-note">
          Each chart shows the model's value minus the baseline's value for
          that season — above 0 means the model was better (MAE is reversed,
          so a positive edge is good there too).
        </p>
        <div className="chart-grid">
          {METRICS.map((metric) => (
            <figure key={metric.key} className="chart-figure">
              <figcaption>{metric.label} — model minus baseline</figcaption>
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
      ariaLabel={`${metricLabel(metric)} edge vs grid and championship, by season`}
      referenceLine={{ y: 0, label: 'even' }}
      valueFormat={(v) => fmtNumber(v, 3)}
    />
  )
}
