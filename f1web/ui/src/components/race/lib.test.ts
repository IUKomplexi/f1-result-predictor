import { describe, expect, it } from 'vitest'
import type { ModelsResponse } from '../../api/client'
import { checkGrid, modelPathFor, racePredictBody, type RaceOverrides } from './lib'

const models: ModelsResponse = {
  default: 'data/model/hurdle.joblib',
  models: {
    alpha: { checkpoint: 'data/model/alpha.joblib' },
    beta: { checkpoint: 'data/model/beta.joblib' },
  },
}

describe('checkGrid', () => {
  it('serializes valid rows to a driver_id,grid CSV', () => {
    const check = checkGrid({ russell: '1', hamilton: ' 2 ' })
    expect(check.error).toBeNull()
    expect(check.csv).toBe('driver_id,grid\nrussell,1\nhamilton,2')
  })

  it('skips empty cells and yields no CSV when nothing is filled', () => {
    const check = checkGrid({ russell: ' ', hamilton: '' })
    expect(check.error).toBeNull()
    expect(check.csv).toBeNull()
  })

  it('rejects non-integer positions and names the driver', () => {
    const check = checkGrid({ russell: '1', hamilton: 'x' })
    expect(check.csv).toBeNull()
    expect(check.error).toContain('hamilton')
    expect(check.error).toContain('positive integer')
  })
})

describe('racePredictBody', () => {
  const base: RaceOverrides = {
    season: 2026,
    round: 5,
    refresh: false,
    model: 'default',
    gridRows: null,
    writeReport: false,
  }

  it('maps the config default to a minimal body without model_path', () => {
    const { body, error } = racePredictBody(models, base)
    expect(error).toBeNull()
    expect(body).toEqual({ season: 2026, round: 5 })
  })

  it('maps a saved model to its checkpoint path', () => {
    const { body } = racePredictBody(models, { ...base, model: 'alpha' })
    expect(body.model_path).toBe('data/model/alpha.joblib')
  })

  it('attaches refresh, write_report and the grid CSV (no feature toggles)', () => {
    const { body, error } = racePredictBody(models, {
      ...base,
      model: 'beta',
      refresh: true,
      writeReport: true,
      gridRows: { russell: '3' },
    })
    expect(error).toBeNull()
    expect(body).toEqual({
      season: 2026,
      round: 5,
      refresh: true,
      write_report: true,
      model_path: 'data/model/beta.joblib',
      grid_csv: 'driver_id,grid\nrussell,3',
    })
    // Feature overrides only make sense at training time (Train tab), so the
    // per-race body never carries them.
    expect(body.enable_features).toBeUndefined()
    expect(body.disable_features).toBeUndefined()
  })

  it('returns a user-facing error for an invalid grid instead of a body to send', () => {
    const { error } = racePredictBody(models, { ...base, gridRows: { russell: 'abc' } })
    expect(error).toContain('russell')
  })
})

describe('modelPathFor', () => {
  it('is undefined for the default choice, unknown names and missing models', () => {
    expect(modelPathFor(models, 'default')).toBeUndefined()
    expect(modelPathFor(models, 'nope')).toBeUndefined()
    expect(modelPathFor(null, 'alpha')).toBeUndefined()
  })

  it('resolves saved model names to their checkpoint paths', () => {
    expect(modelPathFor(models, 'beta')).toBe('data/model/beta.joblib')
  })
})
