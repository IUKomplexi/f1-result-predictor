import { describe, expect, it } from 'vitest'
import { cellTone, METRICS } from './lib'

describe('cellTone', () => {
  it('marks the best value green and the worst red (higher-better metrics)', () => {
    const values = [0.4, 0.6, 0.5]
    expect(cellTone('spearman', values, 0.6)).toBe('best')
    expect(cellTone('spearman', values, 0.4)).toBe('worst')
    expect(cellTone('spearman', values, 0.5)).toBe('mid')
  })

  it('inverts the direction for MAE (lower is better)', () => {
    const values = [3.0, 2.2, 2.8]
    expect(cellTone('mae', values, 2.2)).toBe('best')
    expect(cellTone('mae', values, 3.0)).toBe('worst')
    expect(cellTone('mae', values, 2.8)).toBe('mid')
  })

  it('stays neutral when all values are tied or missing', () => {
    expect(cellTone('mae', [2.5, 2.5, 2.5], 2.5)).toBe('neutral')
    expect(cellTone('mae', [NaN, NaN], NaN)).toBe('neutral')
  })

  it('covers every metric with a label (Top-10 overlap included)', () => {
    const keys = METRICS.map((m) => m.key)
    expect(keys).toContain('top10_overlap')
  })
})
