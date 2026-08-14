import type { Backtest } from '../../api/client'
import { Badge } from '../../components/ui/Badge'
import {
  BASELINES,
  BASELINE_HELP,
  BASELINE_LABEL,
  METRICS,
} from './lib'
import { MetricCell } from './MetricCell'
import { ModelComparison } from './ModelComparison'

/** Plain-language metric definitions under the mean table. */
const METRIC_HELP: Record<string, string> = {
  winner_hit: 'how often the predicted winner actually won',
  top3_overlap: 'share of the actual podium the model named',
  top10_overlap: 'share of the actual top 10 the model named',
  spearman: 'rank correlation of predicted vs actual points order',
  mae: 'mean absolute error of predicted points',
}

/** The persisted backtest report: mean table and model comparison. */
export function BacktestView({
  backtest,
  reference,
  deployedLabel,
}: {
  backtest: Backtest
  reference: string | null
  /** Checkpoint stem of the deployed default model (e.g. "hurdle"). */
  deployedLabel: string
}) {
  return (
    <>
      <section className="card">
        <h2 className="card-title">Model vs baselines (mean)</h2>
        <p className="context-note">
          The <strong>Model</strong> row is always the deployed default model (
          {deployedLabel}), retrained walk-forward per season — it is not one
          of the compared models below.
        </p>
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
                        <Badge variant="info">{BASELINE_LABEL[baseline]} · {deployedLabel}</Badge>
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
        <details className="metric-glossary-toggle">
          <summary>What do these metrics and baselines mean?</summary>
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
        </details>
      </section>

      {backtest.models && Object.keys(backtest.models).length > 1 ? (
        <ModelComparison backtest={backtest} reference={reference} />
      ) : null}

    </>
  )
}
