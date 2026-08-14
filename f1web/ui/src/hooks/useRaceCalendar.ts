import { useEffect, useState } from 'react'
import { getCalendar, getPrediction, type Status as PipelineStatus } from '../api/client'
import type { NavState } from '../App'
import type { LoadState } from './useApi'

/**
 * Race tab state machine: defaults to the global "next race" (falling back to
 * the newest configured season), loads the ordered round list whenever the
 * season changes, and snaps to a valid round. Encapsulates the prime →
 * calendar → snap effect chain so the Race view stays purely declarative.
 *
 * A cross-tab ``initial`` (from Race History's "open this race") overrides
 * the default selection.
 */
export function useRaceCalendar(
  status: LoadState<PipelineStatus>,
  initial: NavState | null | undefined = null,
): {
  season: number | null
  round: number | null
  rounds: number[]
  /** Race names by round (GP name shown in the pager). */
  roundNames: Map<number, string>
  /** The global next race (from /api/prediction), for the quick-jump. */
  nextRace: { season: number; round: number } | null
  /** Why the next-race lookup failed (null when it succeeded or is pending). */
  primeError: string | null
  /** The season shown in the picker (season ?? newest configured season). */
  selected: number | null
  /** Selectable seasons, newest first (a next-race season beyond the
   *  configured range stays selectable). */
  seasons: number[]
  selectSeason: (season: number) => void
  setRound: (round: number | null) => void
  /** Jump to the global next race (switches season when needed). */
  goToNextRace: () => void
} {
  const [season, setSeason] = useState<number | null>(null)
  const [round, setRound] = useState<number | null>(null)
  const [rounds, setRounds] = useState<number[]>([])
  const [roundNames, setRoundNames] = useState<Map<number, string>>(new Map())
  // The season the loaded `rounds` list belongs to. The calendar fetch is
  // async, so `rounds` briefly holds the *previous* season's list after a
  // switch; snapping against it would land on a round that is invalid (or
  // just wrong) for the newly selected season.
  const [roundsSeason, setRoundsSeason] = useState<number | null>(null)
  const [nextRace, setNextRace] = useState<{ season: number; round: number } | null>(null)
  const [primed, setPrimed] = useState(false)
  // When the next-race lookup fails (e.g. a checkpoint/feature mismatch), the
  // reason is surfaced in the Race view instead of silently falling back.
  const [primeError, setPrimeError] = useState<string | null>(null)

  const configuredEnd = status.phase === 'ready' ? status.data.seasons.end : null
  const configuredStart = status.phase === 'ready' ? status.data.seasons.start : null

  // Default to the global "next race" once status is ready — unless a
  // cross-tab navigation state preselects a specific race.
  useEffect(() => {
    if (primed || status.phase !== 'ready') return
    setPrimed(true)
    if (initial?.season !== undefined && initial?.round !== undefined) {
      setSeason(initial.season)
      setRound(initial.round)
      return
    }
    const fallbackSeason = status.data.seasons.end
    getPrediction()
      .then((pred) => {
        setPrimeError(null)
        setSeason(pred.season)
        setRound(pred.round)
        setNextRace({ season: pred.season, round: pred.round })
      })
      .catch((error: unknown) => {
        // Keep the reason visible (e.g. "checkpoint feature set does not
        // match … retrain with model/train.py") and fall back to the newest
        // configured season (no round -> snapped below).
        setPrimeError(error instanceof Error ? error.message : String(error))
        setSeason(fallbackSeason)
        setRound(null)
      })
  }, [primed, status.phase, initial])

  // Load the ordered round list whenever the season changes.
  useEffect(() => {
    if (season === null) return
    let cancelled = false
    getCalendar(season)
      .then((data) => {
        if (cancelled) return
        setRoundsSeason(season)
        setRounds(data.calendar.map((c) => c.round).sort((a, b) => a - b))
        setRoundNames(
          new Map(data.calendar.map((c) => [c.round, c.race_name] as const)),
        )
      })
      .catch(() => {
        if (!cancelled) {
          setRoundsSeason(null)
          setRounds([])
        }
      })
    return () => {
      cancelled = true
    }
  }, [season])

  // Snap to a valid round once this season's calendar has loaded: if none is
  // selected or the current one is no longer in the calendar, fall back to
  // the next race (when it is in this season — the useful default) or the
  // most recent round. Guarded on `roundsSeason` so a stale list from the
  // previously selected season can never be snapped against.
  useEffect(() => {
    if (season === null || roundsSeason !== season || rounds.length === 0) return
    if (round === null || !rounds.includes(round)) {
      const target =
        nextRace !== null && nextRace.season === season
          ? nextRace.round
          : rounds[rounds.length - 1]
      setRound(target)
    }
  }, [season, round, rounds, roundsSeason, nextRace])

  const selected = season ?? configuredEnd
  const configured =
    configuredStart !== null && configuredEnd !== null
      ? Array.from({ length: configuredEnd - configuredStart + 1 }, (_, i) => configuredEnd - i)
      : []
  const seasons =
    selected !== null && !configured.includes(selected)
      ? [selected, ...configured]
      : configured

  const selectSeason = (next: number) => {
    setSeason(next)
    setRound(null)
  }

  const goToNextRace = () => {
    if (nextRace === null) return
    if (nextRace.season !== season) setSeason(nextRace.season)
    setRound(nextRace.round)
  }

  return {
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
  }
}
