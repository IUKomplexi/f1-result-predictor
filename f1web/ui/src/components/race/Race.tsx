import { useEffect, useRef, useState } from 'react'
import {
  getModels,
  getStatus,
  postPrediction,
  type ModelsResponse,
  type Prediction,
} from '../../api/client'
import type { TabProps } from '../../App'
import { useApi } from '../../hooks/useApi'
import { useRaceCalendar } from '../../hooks/useRaceCalendar'
import { driverLabel, fmtDate, fmtPoints } from '../../lib/format'
import { deployedName } from '../backtest/lib'
import { Badge } from '../ui/Badge'
import { ErrorState, Skeleton } from '../ui/DataState'
import { ProbabilityBar } from '../ui/ProbabilityBar'
import { GridEditor } from './GridEditor'
import { ModelPicker, RACE_DEFAULT_MODEL } from './ModelPicker'
import { racePredictBody } from './lib'
import { RaceScoreboard } from './RaceScoreboard'
import './Race.css'

/**
 * Race view: one race's prediction at a time, with a season selector and
 * prev/next round navigation. Defaults to the upcoming "next race"; prior
 * completed rounds show their verified prediction. Accepts a cross-tab
 * navigation state (e.g. from Race History's "open this race").
 */
export function Race({ navState }: TabProps) {
  const status = useApi('status', () => getStatus())
  const modelsState = useApi<ModelsResponse>('models', getModels)
  const models = modelsState.state.phase === 'ready' ? modelsState.state.data : null
  const [model, setModel] = useState<string>(RACE_DEFAULT_MODEL)
  const [defaultModel, setDefaultModel] = useState<string>(RACE_DEFAULT_MODEL)
  const [modelTouched, setModelTouched] = useState(false)
  const {
    season,
    round,
    rounds,
    roundNames,
    nextRace,
    primeError,
    selected,
    seasons,
    selectSeason,
    setRound,
    goToNextRace,
  } = useRaceCalendar(status.state, navState)

  // Preselect the deployed model once the index loads; never clobber a choice
  // the user already made. defaultModel is the baseline the "pending
  // overrides" badge compares against.
  useEffect(() => {
    if (modelTouched || models === null) return
    const deployed = deployedName(models) ?? RACE_DEFAULT_MODEL
    setDefaultModel(deployed)
    setModel(deployed)
  }, [models, modelTouched])

  if (status.state.phase === 'loading') return <Skeleton rows={8} />
  if (status.state.phase === 'error') {
    return <ErrorState message={status.state.message} onRetry={status.retry} />
  }

  const idx = round !== null ? rounds.indexOf(round) : -1
  const canPrev = idx > 0
  const canNext = idx >= 0 && idx < rounds.length - 1
  const isNextRace =
    nextRace !== null && nextRace.season === season && nextRace.round === round

  return (
    <>
      {primeError ? (
        <section className="card race-prime-error" role="alert">
          <p className="save-status error">
            <strong>Prediction unavailable:</strong> {primeError}
          </p>
          <p className="muted">
            The deployed model's feature set does not match the configured
            features — retrain it from the Train tab (leave the model name
            empty to replace the current model).
          </p>
        </section>
      ) : null}
      {season === null || round === null ? (
        <Skeleton rows={10} />
      ) : (
        <RacePanel
          season={season}
          round={round}
          models={models}
          model={model}
          defaultModel={defaultModel}
          onModelChange={(value) => {
            setModelTouched(true)
            setModel(value)
          }}
          nav={{
            selected,
            seasons,
            selectSeason,
            rounds,
            idx,
            canPrev,
            canNext,
            setRound,
            isNextRace,
            nextRace,
            goToNextRace,
            roundNames,
          }}
        />
      )}
    </>
  )
}

/**
 * One race's prediction via POST /api/predict so every per-request CLI
 * override is available: the model checkpoint and a qualifying grid. The
 * report is always written (reports/prediction.md); "refresh from API" lives
 * on the Data tab. Feature overrides are NOT offered here — they only matter
 * at training time (Train tab, or Backtest's walk-forward retraining). Edits
 * are local until "Apply changes"; the request is cached server-side per
 * override combination.
 *
 * The whole top area is one "control deck" card: the race title + meta on
 * the left, season/model pickers and the round pager on the right, and a
 * collapsible drawer for grid overrides below.
 */
type RaceNav = {
  selected: number | null
  seasons: number[]
  selectSeason: (season: number) => void
  rounds: number[]
  idx: number
  canPrev: boolean
  canNext: boolean
  setRound: (round: number) => void
  isNextRace: boolean
  nextRace: { season: number; round: number } | null
  goToNextRace: () => void
  roundNames: Map<number, string>
}

function RacePanel({
  season,
  round,
  models,
  model,
  defaultModel,
  onModelChange,
  nav,
}: {
  season: number
  round: number
  models: ModelsResponse | null
  model: string
  defaultModel: string
  onModelChange: (value: string) => void
  nav: RaceNav
}) {
  const [gridRows, setGridRows] = useState<Record<string, string> | null>(null)
  const [applyError, setApplyError] = useState<string | null>(null)
  const [applied, setApplied] = useState(0)

  // The qualifying grid belongs to one race; switching races drops the edits.
  useEffect(() => {
    setGridRows(null)
  }, [season, round])

  const { state, retry } = useApi(
    `race-${season}-${round}-${applied}`,
    () => {
      const { body } = racePredictBody(models, {
        season,
        round,
        refresh: false,
        model,
        gridRows,
        writeReport: true,
      })
      return postPrediction(body)
    },
  )

  // The grid editor seeds from the latest ready prediction, so it stays
  // populated while a re-request is in flight.
  const lastRef = useRef<Prediction | null>(null)
  if (state.phase === 'ready') lastRef.current = state.data
  const last = lastRef.current

  const apply = () => {
    const { error } = racePredictBody(models, {
      season,
      round,
      refresh: false,
      model,
      gridRows,
      writeReport: true,
    })
    if (error !== null) {
      setApplyError(error)
      return
    }
    setApplyError(null)
    setApplied((n) => n + 1)
  }

  const pendingOverrides = model !== defaultModel || gridRows !== null
  const gpName = nav.roundNames.get(round)
  const title = last?.race.race_name ?? gpName ?? `Round ${round}`
  const meta = last?.race

  return (
    <>
      <section className="card race-deck">
        <div className="race-deck-header">
          <div className="race-title-group">
            <h2 className="card-title">{title}</h2>
            <p className="meta-line">
              <span>Round {round} · season {season}</span>
              {meta ? (
                <>
                  <span>·</span>
                  <span>{fmtDate(meta.date)}</span>
                  {meta.circuit_id ? (
                    <>
                      <span>·</span>
                      <span>{driverLabel(meta.circuit_id)}</span>
                    </>
                  ) : null}
                </>
              ) : null}
            </p>
          </div>

          <div className="race-deck-controls">
            <div className="control-group">
              <label className="field">
                <span className="field-label">Season</span>
                <select
                  className="select"
                  value={nav.selected ?? ''}
                  onChange={(event) => nav.selectSeason(Number(event.target.value))}
                >
                  {nav.seasons.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>
              <ModelPicker models={models} value={model} onChange={onModelChange} />
            </div>
            <div className="pager">
              <span className="pager-label">
                {gpName !== undefined ? `${gpName} · Round ${round}` : `Round ${round}`}
              </span>
              <div className="pager-buttons">
                <button
                  type="button"
                  className="button"
                  disabled={!nav.canPrev}
                  onClick={() => nav.setRound(nav.rounds[nav.idx - 1])}
                >
                  ‹ Prev
                </button>
                <button
                  type="button"
                  className="button"
                  disabled={!nav.canNext}
                  onClick={() => nav.setRound(nav.rounds[nav.idx + 1])}
                >
                  Next ›
                </button>
                {nav.isNextRace ? (
                  <Badge variant="info">Upcoming</Badge>
                ) : (
                  <button
                    type="button"
                    className="button"
                    disabled={nav.nextRace === null}
                    onClick={nav.goToNextRace}
                    title={
                      nav.nextRace !== null
                        ? `Jump to the next race (round ${nav.nextRace.round})`
                        : undefined
                    }
                  >
                    Next race
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>

        <details className="advanced-options advanced-options-inline">
          <summary>Grid &amp; model overrides</summary>
          <div className="advanced-drawer">
            {last !== null && !last.verified ? (
              <GridEditor
                drivers={last.drivers}
                values={gridRows}
                onChange={setGridRows}
                onReset={() => setGridRows(null)}
              />
            ) : (
              <p className="muted">
                Grid overrides only apply to an upcoming race with no results
                yet.
              </p>
            )}
            <div className="drawer-actions">
              {pendingOverrides ? (
                <Badge variant="warn">
                  {gridRows !== null && model !== defaultModel
                    ? 'Grid + model overrides pending'
                    : gridRows !== null
                      ? 'Grid override pending'
                      : 'Model override pending'}
                </Badge>
              ) : (
                <span className="muted">Config defaults</span>
              )}
              {applyError ? (
                <p className="save-status error" role="alert">
                  {applyError}
                </p>
              ) : null}
              <button
                type="button"
                className="button primary"
                onClick={apply}
                disabled={state.phase === 'loading'}
              >
                Apply changes
              </button>
            </div>
          </div>
        </details>
      </section>
      {state.phase === 'loading' ? <Skeleton rows={10} /> : null}
      {state.phase === 'error' ? (
        <ErrorState message={state.message} onRetry={retry} />
      ) : null}
      {state.phase === 'ready' ? <RaceTable prediction={state.data} /> : null}
    </>
  )
}

function RaceTable({ prediction }: { prediction: Prediction }) {
  const { drivers, synthetic, verified, calibrated, checkpoint } = prediction
  const modelStem = checkpoint.split(/[\\/]/).pop()?.replace(/\.joblib$/, '') ?? checkpoint
  return (
    <>
      {verified ? <RaceScoreboard drivers={drivers} /> : null}

      <section className="card">
        <div className="race-table-head">
          <h2 className="card-title">Ranked grid</h2>
          <div className="badge-row">
            <span className="model-chip" title={checkpoint}>
              {modelStem}
            </span>
            {synthetic ? (
              <Badge variant="warn">Unverified · synthetic grid</Badge>
            ) : null}
            {verified ? <Badge variant="ready">Has actuals</Badge> : null}
            {calibrated ? <Badge variant="info">Calibrated probabilities</Badge> : null}
          </div>
        </div>
        <div className="table-wrap table-scroll">
          <table className="data-table grid-table">
            <thead>
              <tr>
                <th scope="col" className="num">#</th>
                <th scope="col">Driver</th>
                <th scope="col" className="hide-narrow">Team</th>
                <th scope="col" className="num">Grid</th>
                <th scope="col" className="num">Exp. pts</th>
                <th scope="col" className="num">P scored</th>
                <th scope="col" className="num">P top 3</th>
                <th scope="col" className="num">P win</th>
                {verified ? <th scope="col" className="num">Actual</th> : null}
                {verified ? <th scope="col" className="num">Δ pts</th> : null}
              </tr>
            </thead>
            <tbody>
              {drivers.map((row) => (
                <tr key={row.driver_id}>
                  <td className="num rank">{row.pred_rank}</td>
                  <td className="driver">{driverLabel(row.driver_id)}</td>
                  <td className="muted hide-narrow">{driverLabel(row.constructor_id)}</td>
                  <td className="num">{row.grid ?? '–'}</td>
                  <td className="num">{fmtPoints(row.expected_points)}</td>
                  <td className="num">
                    <ProbabilityBar value={row.p_scored} label="P scored" variant="scored" />
                  </td>
                  <td className="num">
                    <ProbabilityBar value={row.p_top3} label="P top 3" variant="top3" />
                  </td>
                  <td className="num">
                    <ProbabilityBar value={row.p_win} label="P win" variant="win" />
                  </td>
                  {verified ? (
                    <td className="num">
                      {row.actual_position ? `P${row.actual_position}` : '–'} ·{' '}
                      {fmtPoints(row.actual_points)}
                    </td>
                  ) : null}
                  {verified ? (
                    <td className="num">
                      <PointsDelta expected={row.expected_points} actual={row.actual_points} />
                    </td>
                  ) : null}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {verified ? (
          <p className="muted delta-legend">
            Δ pts = actual − expected · green: model under-predicted, amber: over-predicted.
          </p>
        ) : null}
      </section>
    </>
  )
}

/** Colored actual − expected points delta (green under-predicted / amber over-predicted). */
function PointsDelta({ expected, actual }: { expected: number; actual: number | null }) {
  if (actual === null || actual === undefined) return <span className="muted">–</span>
  const delta = actual - expected
  const cls =
    Math.abs(delta) < 0.05 ? 'delta-zero' : delta > 0 ? 'delta-pos' : 'delta-neg'
  return (
    <span className={cls} title="actual − expected points">
      {delta > 0 ? '+' : ''}
      {fmtPoints(delta)}
    </span>
  )
}
