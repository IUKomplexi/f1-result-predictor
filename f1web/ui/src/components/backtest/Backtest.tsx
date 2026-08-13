import { useState } from 'react'
import { getBacktest, getStatus, type Backtest, type BacktestMetricRow } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { fmtNumber } from '../../lib/format'
import { Badge } from '../ui/Badge'
import { Chart, type ChartDatum, type ChartSeries } from '../ui/Chart'
import { ErrorState, Skeleton } from '../ui/DataState'
import { FeatureToggles, NO_FEATURE_OVERRIDES, type FeatureOverride } from '../ui/FeatureToggles'
import { JobRunner } from '../ui/JobRunner'
import { ModelPicker, modelCheckpointPath, useModels, type ModelSelection } from '../ui/ModelPicker'
import { PrereqHint } from '../ui/PrereqHint'
import { RefreshToggle } from '../ui/RefreshToggle'
import {
  DEFAULT_SEASON_RANGE,
  SeasonRange,
  seasonPayload,
  type SeasonRangeValue,
} from '../ui/SeasonRange'
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

/**
 * Backtest: pick a trained model and score every selected season with it
 * (using the features it was trained on) — the "how good is THIS model" view.
 * Walk-forward retraining and output quantization live under Advanced.
 */
export function Backtest() {
  const [modelChoice, setModelChoice] = useState<ModelSelection>('default')
  const [walkForward, setWalkForward] = useState(false)
  const [quantize, setQuantize] = useState(true)
  const [range, setRange] = useState<SeasonRangeValue>(DEFAULT_SEASON_RANGE)
  const [refresh, setRefresh] = useState(false)
  const [features, setFeatures] = useState<FeatureOverride>(NO_FEATURE_OVERRIDES)
  const [version, setVersion] = useState(0)
  const status = useApi('status', () => getStatus())
  const { state: modelsState } = useModels()
  const models = modelsState.phase === 'ready' ? modelsState.data : null
  const { state, retry } = useApi(`backtest-${version}`, () => getBacktest())
  return (
    <>
      <JobRunner
        type="backtest"
        runLabel="Run backtest"
        onDone={() => setVersion((v) => v + 1)}
        buildPayload={() => ({
          ...seasonPayload(range),
          refresh,
          quantize,
          // With a model chosen (or the config default), score with that fixed
          // model. Walk-forward retraining is the explicit advanced opt-in.
          use_checkpoint: !walkForward,
          model_path: walkForward ? null : modelCheckpointPath(models, modelChoice),
          enable_features: walkForward ? features.enable : [],
          disable_features: walkForward ? features.disable : [],
        })}
        options={
          <>
            <ModelPicker
              value={modelChoice}
              onChange={setModelChoice}
              disabled={walkForward}
            />
            <SeasonRange value={range} onChange={setRange} />
            <RefreshToggle value={refresh} onChange={setRefresh} />
            <details className="advanced-options">
              <summary>Advanced</summary>
              <div className="job-options-inner">
                <div className="job-option">
                  <label
                    className="check-line"
                    title="Ignore the selected model and retrain on every test season (train = all strictly earlier seasons)."
                  >
                    <input
                      type="checkbox"
                      checked={walkForward}
                      onChange={(e) => setWalkForward(e.target.checked)}
                    />
                    Walk-forward retraining
                  </label>
                  <p className="job-option-hint">
                    Honest out-of-sample estimates, but does not tell you how
                    good the model you just trained is.
                  </p>
                </div>
                <div className="job-option">
                  <label
                    className="check-line"
                    title="Round expected points to the nearest points-table value (matches the deployed predictor)."
                  >
                    <input
                      type="checkbox"
                      checked={quantize}
                      onChange={(e) => setQuantize(e.target.checked)}
                    />
                    Quantize points
                  </label>
                </div>
                {walkForward ? (
                  <FeatureToggles value={features} onChange={setFeatures} />
                ) : null}
              </div>
            </details>
          </>
        }
        renderResult={(job) => <BacktestRunResult job={job} />}
      />
      <PrereqHint
        when={status.state.phase === 'ready' && !status.state.data.model.has_checkpoint}
      >
        No model checkpoint yet — run Train first so there is a model to
        score.
      </PrereqHint>
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
  const checkpoint = job.result?.checkpoint as string | undefined
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
      {checkpoint ? (
        <p className="summary-list">
          Model: <code className="mono">{checkpoint}</code>
        </p>
      ) : null}
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

// Metrics where a higher value is better. For the rest (MAE), lower is better,
// so the edge is computed as baseline - model to keep "positive = model wins".
const IMPROVE_UP = new Set<keyof BacktestMetricRow>([
  'winner_hit',
  'top3_overlap',
  'spearman',
])

function edge(
  model: number | undefined,
  baseline: number | undefined,
  metric: keyof BacktestMetricRow,
): number | undefined {
  if (model === undefined || baseline === undefined) return undefined
  return IMPROVE_UP.has(metric) ? model - baseline : baseline - model
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
      vsGrid: edge(bySeason.model?.[s]?.[metric], bySeason.grid?.[s]?.[metric], metric),
      vsChamp: edge(
        bySeason.model?.[s]?.[metric],
        bySeason.championship?.[s]?.[metric],
        metric,
      ),
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
