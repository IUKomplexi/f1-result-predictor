import { describe, expect, it } from 'vitest'
import { DEFAULT_SEASON_RANGE, resolveRange, seasonPayload } from './SeasonRange'

const seasons = { start: 2010, end: 2026, data_start: 2014, data_end: 2026 }

describe('resolveRange', () => {
  it('fills blank inputs with the allowed window (same as the pickers)', () => {
    expect(resolveRange(DEFAULT_SEASON_RANGE, seasons)).toEqual({ start: 2014, end: 2026 })
  })

  it('keeps explicit values unchanged', () => {
    expect(resolveRange({ start: 2016, end: 2025 }, seasons)).toEqual({ start: 2016, end: 2025 })
  })

  it('fills only the side the user left blank', () => {
    expect(resolveRange({ start: 2016, end: null }, seasons)).toEqual({ start: 2016, end: 2026 })
    expect(resolveRange({ start: null, end: 2020 }, seasons)).toEqual({ start: 2014, end: 2020 })
  })

  it('returns the raw range unchanged before status has loaded', () => {
    expect(resolveRange(DEFAULT_SEASON_RANGE, null)).toEqual(DEFAULT_SEASON_RANGE)
  })
})

describe('seasonPayload', () => {
  it('omits blank fields so the backend applies its own defaults', () => {
    expect(seasonPayload(DEFAULT_SEASON_RANGE)).toEqual({})
    expect(seasonPayload({ start: 2016, end: null })).toEqual({ start: 2016 })
    expect(seasonPayload({ start: 2014, end: 2026 })).toEqual({ start: 2014, end: 2026 })
  })
})
