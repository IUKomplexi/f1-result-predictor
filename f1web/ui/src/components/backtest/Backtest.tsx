import { useEffect, useRef, useState } from 'react'
import {
  getBacktest,
  getModels,
  getStatus,
  type Backtest,
  type BacktestMetricRow,
  type ModelInfo,
  type ModelsResponse,
} from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { fmtDate, fmtNumber } from '../../lib/format'
import { Badge } from '../ui/Badge'
import { Chart, type ChartDatum, type ChartSeries } from '../ui/Chart'
import { ErrorState, Skeleton } from '../ui/DataState'
import { FeatureToggles, NO_FEATURE_OVERRIDES, type FeatureOverride } from '../ui/FeatureToggles'
import { JobRunner } from '../ui/JobRunner'
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

/** Palette for per-model comparison series (baselines keep their own colors). */
const MODEL_COLORS = ['#e10600', '#4a7fd6', '#d9a514', '#17a354', '#8b5cf6', '#0ea5e9', '#f97316']

const METRICS: { key: keyof BacktestMetricRow; label: string }[] = [
  { key: 'winner_hit', label: 'Winner hit' },
  { key: 'top3_overlap', label: 'Top-3 overlap' },
  { key: 'spearman', label: 'Spearman' },
  { key: 'mae', label: 'MAE' },
]

/** The "deployed config model" pseudo-choice, only when it is not a saved model. */
const DEFAULT_MODEL = 'default'

/** Normalize a checkpoint path for identity comparison (Windows separators). */
function normPath(path: string | null | undefined): string | null {
  if (!path) return null
  return path.replace(/\\/g, '/').toLowerCase()
}

interface ModelChoice {
  /** Selection value: a saved model name, or DEFAULT_MODEL for the config default. */
  value: string
  label: string
}

/** The saved model whose checkpoint equals the config default, if any. */
function deployedName(models: ModelsResponse | null): string | null {
  if (!models) return null
  const target = normPath(models.default)
  if (!target) return null
  for (const [name, info] of Object.entries(models.models)) {
    if (normPath(info.checkpoint) === target) return name
  }
  return null
}

function modelChoices(models: ModelsResponse | null): ModelChoice[] {
  const saved = models ? Object.keys(models.models).sort() : []
  const deployed = deployedName(models)
  const choices: ModelChoice[] = saved.map((name) => ({
    value: name,
    label: deployed === name ? `${name} (deployed)` : name,
  }))
  if (!models || deployed !== null) return choices
  // The config default points at a checkpoint that is not in the saved index
  // (CLI-trained, or the [model] checkpoint was edited in Settings): keep a
  // pseudo-entry so that model stays selectable. When it IS a saved model we
  // skip this — that model is already listed once, marked (deployed).
  return [{ value: DEFAULT_MODEL, label: 'config default (deployed)' }, ...choices]
}

function selectedPaths(models: ModelsResponse | null, checked: string[]): string[] {
  if (!models) return []
  const seen = new Set<string>()
  const paths: string[] = []
  for (const value of checked) {
    const path = value === DEFAULT_MODEL ? models.default : models.models[value]?.checkpoint
    if (typeof path !== 'string' || path.length === 0) continue
    const key = normPath(path)
    if (key === null || seen.has(key)) continue
    seen.add(key)
    paths.push(path)
  }
  return paths
}

function defaultStem(models: ModelsResponse | null): string | null {
  if (!models) return null
  return models.default.split(/[\\/]/).pop()?.replace(/\.joblib$/, '') ?? null
}

/**
 * Backtest: score every selected season with one or several saved models
 * (each using the features it was trained on) and compare them against the
 * grid / championship / zero baselines — the "how good is THIS model" view.
 * Two or more selected models additionally produce per-metric comparison
 * charts. Walk-forward retraining and output quantization live under
 * Advanced.
 */
export function Backtest() {
  const [checked, setChecked] = useState<string[]>([])
  const [walkForward, setWalkForward] = useState(false)
  const [quantize, setQuantize] = useState(true)
  const [range, setRange] = useState<SeasonRangeValue>(DEFAULT_SEASON_RANGE)
  const [refresh, setRefresh] = useState(false)
  const [features, setFeatures] = useState<FeatureOverride>(NO_FEATURE_OVERRIDES)
  const [version, setVersion] = useState(0)
  const status = useApi('status', () => getStatus())
  const modelsState = useApi<ModelsResponse>('models', getModels)
  const models = modelsState.state.phase === 'ready' ? modelsState.state.data : null
  const { state, retry } = useApi(`backtest-${version}`, () => getBacktest())
  const choices = modelChoices(models)
  const deployed = deployedName(models)
  // Guards the default selection: once the model index loads, preselect the
  // deployed model, but never clobber a selection the user made themselves.
  const userTouched = useRef(false)

  useEffect(() => {
    if (userTouched.current || models === null) return
    setChecked([deployed ?? DEFAULT_MODEL])
  }, [models, deployed])

  function toggleModel(value: string, on: boolean) {
    userTouched.current = true
    setChecked((current) => (on ? [...current, value] : current.filter((v) => v !== value)))
  }

  return (
    <>
      <ModelsOverview models={models} />
      <JobRunner
        type="backtest"
        runLabel="Run backtest"
        onDone={() => setVersion((v) => v + 1)}
        buildPayload={() => ({
          ...seasonPayload(range),
          refresh,
          quantize,
          // Walk-forward retraining is the explicit advanced opt-in; otherwise
          // the selected saved models are scored with their own features.
          use_checkpoint: !walkForward,
          ...(walkForward
            ? { enable_features: features.enable, disable_features: features.disable }
            : { model_paths: selectedPaths(models, checked) }),
        })}
        options={
          <>
            <div className="job-option model-select">
              <span className="job-label">Models to score</span>
              <div className="model-check-list">
                {choices.map((choice) => (
                  <label key={choice.value} className="check-line">
                    <input
                      type="checkbox"
                      checked={checked.includes(choice.value)}
                      disabled={walkForward}
                      onChange={(e) => toggleModel(choice.value, e.target.checked)}
                    />
                    {choice.label}
                  </label>
                ))}
              </div>
              <p className="job-option-hint">
                Each selected model scores the same seasons with its own features. Pick two
                or more to get comparison charts below; uncheck all to score the deployed
                checkpoint via the config feature set.
              </p>
            </div>
            <SeasonRange value={range} onChange={setRange} />
            <RefreshToggle value={refresh} onChange={setRefresh} />
            <details className="advanced-options">
              <summary>Advanced</summary>
              <div className="job-options-inner">
                <div className="job-option">
                  <label
                    className="check-line"
                    title="Ignore the selected models and retrain on every test season (train = all strictly earlier seasons)."
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
        <BacktestView backtest={state.data} reference={defaultStem(models)} />
      )}
    </>
  )
}

/** Saved checkpoints at a glance (name, training window, rows, features). */
function ModelsOverview({ models }: { models: ModelsResponse | null }) {
  const entries = models
    ? Object.entries(models.models).sort(([a], [b]) => a.localeCompare(b))
    : []
  const deployed = deployedName(models)
  return (
    <section className="card">
      <h2 className="card-title">Saved models</h2>
      {models === null ? (
        <Skeleton rows={2} />
      ) : entries.length === 0 ? (
        <p className="muted">No saved models yet — name one on the Train tab.</p>
      ) : (
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Seasons</th>
                <th scope="col" className="num">Rows</th>
                <th scope="col" className="num">Features</th>
                <th scope="col" className="num">Params</th>
                <th scope="col">Trained</th>
              </tr>
            </thead>
            <tbody>
              {entries.map(([name, info]) => (
                <ModelRow key={name} name={name} info={info} deployed={deployed === name} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

function ModelRow({ name, info, deployed }: { name: string; info: ModelInfo; deployed: boolean }) {
  const trainedAt =
    typeof info.trained_at === 'number' ? fmtDate(new Date(info.trained_at * 1000).toISOString()) : '–'
  return (
    <tr className={deployed ? 'row-model' : undefined}>
      <td>
        {name}
        {deployed ? <Badge variant="info">deployed</Badge> : null}
      </td>
      <td>{info.season_range ? `${info.season_range[0]}–${info.season_range[1]}` : '–'}</td>
      <td className="num">{info.rows ?? '–'}</td>
      <td className="num">{info.features?.length ?? '–'}</td>
      <td className="num">{info.params ? Object.keys(info.params).length : '–'}</td>
      <td className="muted">{trainedAt}</td>
    </tr>
  )
}

function BacktestRunResult({ job }: { job: { result: Record<string, unknown> | null; log: string[] } }) {
  const overall = (job.result?.overall ?? {}) as Record<string, Record<string, number>>
  const checkpoint = job.result?.checkpoint as string | undefined
  const compared = (job.result?.models ?? []) as string[]
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
      {compared.length > 1 ? (
        <p className="summary-list">
          Compared models: <code className="mono">{compared.join(', ')}</code>
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

function BacktestView({ backtest, reference }: { backtest: Backtest; reference: string | null }) {
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

/** Per-metric charts with one series per compared model plus the baselines. */
function ModelComparison({ backtest, reference }: { backtest: Backtest; reference: string | null }) {
  const byModel = backtest.models ?? {}
  const names = Object.keys(byModel).sort()
  const refName = reference && names.includes(reference) ? reference : names[0]
  return (
    <section className="card">
      <h2 className="card-title">Model comparison</h2>
      <p className="context-note">
        Each saved model scores the same seasons with its own feature set — the
        baselines are identical across models, so only the model lines differ.
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

// Metrics where a higher value is better. For the rest (MAE), lower is better,
// so the delta is computed as reference - model to keep "positive = better".
const IMPROVE_UP = new Set<keyof BacktestMetricRow>([
  'winner_hit',
  'top3_overlap',
  'spearman',
])

function deltaVs(
  model: BacktestMetricRow | undefined,
  reference: BacktestMetricRow | undefined,
  metric: keyof BacktestMetricRow,
): number | null {
  const a = model?.[metric]
  const b = reference?.[metric]
  if (a === undefined || b === undefined) return null
  return IMPROVE_UP.has(metric) ? a - b : b - a
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
