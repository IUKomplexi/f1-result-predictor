import { act, renderHook, waitFor } from '@testing-library/preact'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { Calendar, Prediction, Status } from '../api/client'
import { getCalendar, getPrediction } from '../api/client'
import type { LoadState } from './useApi'
import { useRaceCalendar } from './useRaceCalendar'

vi.mock('../api/client', () => ({
  getCalendar: vi.fn(),
  getPrediction: vi.fn(),
}))

const status: LoadState<Status> = {
  phase: 'ready',
  data: {
    seasons: { start: 2014, end: 2026, data_start: 2014, data_end: 2026 },
    model: {
      checkpoint: 'data/model/hurdle.joblib',
      calibrators: 'data/model/calibrators.joblib',
      has_checkpoint: true,
      has_calibrators: true,
    },
    data: { dataset: 'data/features.parquet', has_dataset: true, has_raw_cache: true },
    reports: { has_backtest: true, has_calibration: true },
    dashboard: { built: true },
  },
}

function calendar(season: number, rounds: number[]): Calendar {
  return {
    season,
    calendar: rounds.map((round) => ({
      season,
      round,
      race_name: `Round ${round}`,
      date: '2026-01-01',
      time: null,
      circuit_id: null,
      circuit_name: null,
      country: null,
      circuit_lat: null,
      circuit_long: null,
      is_sprint_round: false,
    })),
  }
}

// 2026 has 23 rounds (mid-season, next race = R12); 2025 has 24 (complete).
const CALENDAR_2026 = calendar(2026, Array.from({ length: 23 }, (_, i) => i + 1))
const CALENDAR_2025 = calendar(2025, Array.from({ length: 24 }, (_, i) => i + 1))
const NEXT_RACE = { season: 2026, round: 12 } as Prediction

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((r) => (resolve = r))
  return { promise, resolve }
}

beforeEach(() => {
  vi.mocked(getCalendar).mockReset()
  vi.mocked(getPrediction).mockReset()
})

describe('useRaceCalendar', () => {
  it('primes to the global next race once status is ready', async () => {
    vi.mocked(getPrediction).mockResolvedValue(NEXT_RACE)
    vi.mocked(getCalendar).mockResolvedValue(CALENDAR_2026)
    const { result } = renderHook(() => useRaceCalendar(status))
    await waitFor(() => expect(result.current.season).toBe(2026))
    expect(result.current.round).toBe(12)
    expect(result.current.nextRace).toEqual({ season: 2026, round: 12 })
  })

  it('snaps to the last round when switching to a past season', async () => {
    vi.mocked(getPrediction).mockResolvedValue(NEXT_RACE)
    vi.mocked(getCalendar).mockResolvedValue(CALENDAR_2026)
    const { result } = renderHook(() => useRaceCalendar(status))
    await waitFor(() => expect(result.current.round).toBe(12))

    vi.mocked(getCalendar).mockResolvedValue(CALENDAR_2025)
    act(() => result.current.selectSeason(2025))
    await waitFor(() => expect(result.current.season).toBe(2025))
    await waitFor(() => expect(result.current.round).toBe(24)) // last round, has data
  })

  it('switching back to the current season lands on the next race, never a stale round', async () => {
    vi.mocked(getPrediction).mockResolvedValue(NEXT_RACE)
    vi.mocked(getCalendar).mockResolvedValue(CALENDAR_2026)
    const { result } = renderHook(() => useRaceCalendar(status))
    await waitFor(() => expect(result.current.round).toBe(12))

    // 2025 has 24 rounds, 2026 only 23: the old bug kept round 24 (2025's
    // last) when switching back, which is invalid for 2026 and 409s.
    vi.mocked(getCalendar).mockResolvedValue(CALENDAR_2025)
    act(() => result.current.selectSeason(2025))
    await waitFor(() => expect(result.current.round).toBe(24))

    vi.mocked(getCalendar).mockResolvedValue(CALENDAR_2026)
    act(() => result.current.selectSeason(2026))
    await waitFor(() => expect(result.current.round).toBe(12)) // next race, not 24
    expect(result.current.rounds.includes(result.current.round as number)).toBe(true)
  })

  it('surfaces the next-race lookup error instead of hiding it', async () => {
    vi.mocked(getPrediction).mockRejectedValue(
      new Error(
        "checkpoint feature set does not match the requested feature set; retrain with model/train.py",
      ),
    )
    vi.mocked(getCalendar).mockResolvedValue(CALENDAR_2026)
    const { result } = renderHook(() => useRaceCalendar(status))
    await waitFor(() => expect(result.current.season).toBe(2026))
    await waitFor(() => expect(result.current.primeError).toContain('retrain with model/train.py'))
    // Falls back to the newest configured season (round snaps after calendar).
    await waitFor(() => expect(result.current.round).toBe(23))
    expect(result.current.nextRace).toBeNull()
  })

  it('never snaps against a stale calendar list still in flight', async () => {
    vi.mocked(getPrediction).mockResolvedValue(NEXT_RACE)
    vi.mocked(getCalendar).mockResolvedValue(CALENDAR_2026)
    const { result } = renderHook(() => useRaceCalendar(status))
    await waitFor(() => expect(result.current.round).toBe(12))

    const d2025 = deferred<Calendar>()
    const d2026 = deferred<Calendar>()
    vi.mocked(getCalendar).mockImplementation((s) => {
      if (s === 2025) return d2025.promise
      if (s === 2026) return d2026.promise
      return Promise.resolve(CALENDAR_2026)
    })

    // Switch to 2025 while its calendar is still loading: `rounds` still
    // holds 2026's list, so without the guard the snap would pick round 23
    // from the wrong season.
    act(() => result.current.selectSeason(2025))
    expect(result.current.round).toBeNull()

    // Switch back to 2026; the snap runs only when 2026's own calendar
    // resolves, landing on the next race.
    act(() => result.current.selectSeason(2026))
    act(() => d2026.resolve(CALENDAR_2026))
    await waitFor(() => expect(result.current.round).toBe(12))

    // A late-resolving 2025 calendar must be ignored (its effect was cancelled).
    act(() => d2025.resolve(CALENDAR_2025))
    expect(result.current.round).toBe(12)
    expect(result.current.rounds).toEqual(Array.from({ length: 23 }, (_, i) => i + 1))
  })
})
