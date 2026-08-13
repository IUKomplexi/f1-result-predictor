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
  const {
    season,
    round,
    rounds,
    roundNames,
    nextRace,
    selected,
    seasons,
    selectSeason,
    setRound,
    goToNextRace,
  } = useRaceCalendar(status.state, navState)

  if (status.state.phase === 'loading') return <Skeleton rows={8} />
  if (status.state.phase === 'error') {
    return <ErrorState message={status.state.message} onRetry={status.retry} />
  }

  const idx = round !== null ? rounds.indexOf(round) : -1
  const canPrev = idx > 0
  const canNext = idx >= 0 && idx < rounds.length - 1
  const isNextRace =
    nextRace !== null && nextRace.season === season && nextRace.round === round
  const gpName = round !== null ? roundNames.get(round) : undefined

  return (
    <>
      <section className="card">
        <div className="race-nav">
          <label className="field">
            <span className="field-label">Season</span>
            <select
              className="select"
              value={selected ?? ''}
              onChange={(event) => selectSeason(Number(event.target.value))}
            >
              {seasons.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <div className="pager">
            <button
              type="button"
              className="button"
              disabled={!canPrev}
              onClick={() => setRound(rounds[idx - 1])}
            >
              ‹ Prev
            </button>
            <span className="pager-label">
              {round !== null
                ? `${gpName ?? 'Race'} · Round ${round}`
                : '—'}
            </span>
            <button
              type="button"
              className="button"
              disabled={!canNext}
              onClick={() => setRound(rounds[idx + 1])}
            >
              Next ›
            </button>
            {isNextRace ? (
              <Badge variant="info">Upcoming</Badge>
            ) : (
              <button
                type="button"
                className="button"
                disabled={nextRace === null}
                onClick={goToNextRace}
                title={
                  nextRace !== null
                    ? `Jump to the next race (round ${nextRace.round})`
                    : undefined
                }
              >
                Next race
              </button>
            )}
          </div>
        </div>
      </section>

      {season === null || round === null ? (
        <Skeleton rows={10} />
      ) : (
        <RacePanel season={season} round={round} />
      )}
    </>
  )
}

/**
 * One race's prediction via POST /api/predict so every per-request CLI
 * override is available: model checkpoint, a qualifying grid, refresh and
 * report writing. Feature overrides are NOT offered here — they only matter
 * at training time (Train tab). Edits are local until "Apply changes"; the
 * request is cached server-side per override combination.
 */
function RacePanel({ season, round }: { season: number; round: number }) {
  const [refresh, setRefresh] = useState(false)
  const [model, setModel] = useState<string>(RACE_DEFAULT_MODEL)
  const [defaultModel, setDefaultModel] = useState<string>(RACE_DEFAULT_MODEL)
  const [modelTouched, setModelTouched] = useState(false)
  const [writeReport, setWriteReport] = useState(false)
  const [gridRows, setGridRows] = useState<Record<string, string> | null>(null)
  const [applyError, setApplyError] = useState<string | null>(null)
  const [applied, setApplied] = useState(0)
  const modelsState = useApi<ModelsResponse>('models', getModels)
  const models = modelsState.state.phase === 'ready' ? modelsState.state.data : null

  // Preselect the deployed model once the index loads; never clobber a choice
  // the user already made. defaultModel is the baseline the "pending
  // overrides" badge compares against.
  useEffect(() => {
    if (modelTouched || models === null) return
    const deployed = deployedName(models) ?? RACE_DEFAULT_MODEL
    setDefaultModel(deployed)
    setModel(deployed)
  }, [models, modelTouched])

  // The qualifying grid belongs to one race; switching races drops the edits.
  useEffect(() => {
    setGridRows(null)
  }, [season, round])

  const { state, retry } = useApi(
    `race-${season}-${round}-${applied}-${refresh}`,
    () => {
      const { body } = racePredictBody(models, {
        season,
        round,
        refresh,
        model,
        gridRows,
        writeReport,
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
      refresh,
      model,
      gridRows,
      writeReport,
    })
    if (error !== null) {
      setApplyError(error)
      return
    }
    setApplyError(null)
    setApplied((n) => n + 1)
  }

  const pendingOverrides = model !== defaultModel || gridRows !== null

  return (
    <>
      <section className="card race-controls">
        <div className="race-controls-row">
          <ModelPicker
            models={models}
            value={model}
            onChange={(value) => {
              setModelTouched(true)
              setModel(value)
            }}
          />
          <div className="race-toggles">
            <label
              className="check-line"
              title="Download fresh data instead of using the saved copy."
            >
              <input
                type="checkbox"
                checked={refresh}
                onChange={(e) => setRefresh(e.target.checked)}
              />
              Re-fetch from API (ignore cache)
            </label>
            <label
              className="check-line"
              title="Save the result as reports/prediction.md."
            >
              <input
                type="checkbox"
                checked={writeReport}
                onChange={(e) => setWriteReport(e.target.checked)}
              />
              Write report
            </label>
          </div>
        </div>
        <details className="advanced-options">
          <summary>Advanced</summary>
          <div className="job-options-inner">
            {last !== null && !last.verified ? (
              <GridEditor
                drivers={last.drivers}
                values={gridRows}
                onChange={setGridRows}
                onReset={() => setGridRows(null)}
              />
            ) : null}
          </div>
        </details>
        <div className="race-apply-row">
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
  const { race, drivers, synthetic, verified, calibrated, checkpoint } = prediction
  const modelStem = checkpoint.split(/[\\/]/).pop()?.replace(/\.joblib$/, '') ?? checkpoint
  return (
    <>
      <section className="card">
        <div className="race-meta">
          <div>
            <h2 className="card-title">
              {race.race_name ?? `Round ${prediction.round}`}
            </h2>
            <p className="meta-line">
              <span>
                Round {prediction.round} · season {prediction.season}
              </span>
              <span>·</span>
              <span>{fmtDate(race.date)}</span>
              {race.circuit_id ? (
                <>
                  <span>·</span>
                  <span>{driverLabel(race.circuit_id)}</span>
                </>
              ) : null}
            </p>
          </div>
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
      </section>

      {verified ? <RaceScoreboard drivers={drivers} /> : null}

      <section className="card">
        <h2 className="card-title">Ranked grid</h2>
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
