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
  const [nextRace, setNextRace] = useState<{ season: number; round: number } | null>(null)
  const [primed, setPrimed] = useState(false)

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
        setSeason(pred.season)
        setRound(pred.round)
        setNextRace({ season: pred.season, round: pred.round })
      })
      .catch(() => {
        // Fall back to the newest configured season (no round -> snapped below).
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
        setRounds(data.calendar.map((c) => c.round).sort((a, b) => a - b))
        setRoundNames(
          new Map(data.calendar.map((c) => [c.round, c.race_name] as const)),
        )
      })
      .catch(() => {
        if (!cancelled) setRounds([])
      })
    return () => {
      cancelled = true
    }
  }, [season])

  // Snap to a valid round: if none is selected or the current one is no longer
  // in this season's calendar, fall back to the most recent round.
  useEffect(() => {
    if (season === null || rounds.length === 0) return
    if (round === null || !rounds.includes(round)) {
      setRound(rounds[rounds.length - 1])
    }
  }, [season, round, rounds])

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
    selected,
    seasons,
    selectSeason,
    setRound,
    goToNextRace,
  }
}
