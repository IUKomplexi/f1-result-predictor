import { render, screen, within } from '@testing-library/preact'
import { describe, expect, it } from 'vitest'
import { ModelsOverview } from './ModelsOverview'

const models = {
  models: {
    hurdle: {
      checkpoint: 'data/model/hurdle.joblib',
      params: {
        max_iter: 200,
        learning_rate: 0.05,
        max_depth: 3,
        l2_regularization: 10.0,
        min_samples_leaf: 50,
      },
    },
    legacy: { checkpoint: 'data/model/legacy.joblib' },
  },
  default: 'data/model/hurdle.joblib',
}

describe('ModelsOverview', () => {
  it('renders the trained-on params of each saved model', () => {
    render(<ModelsOverview models={models} />)
    expect(screen.getByText(/max_iter=200/)).toBeDefined()
    expect(screen.getByText(/learning_rate=0\.05/)).toBeDefined()
    expect(screen.getByText(/l2_regularization=10/)).toBeDefined()
    expect(screen.getByText(/min_samples_leaf=50/)).toBeDefined()
  })
  it('marks the deployed model; models without params fall back to an em dash', () => {
    render(<ModelsOverview models={models} />)
    expect(screen.getByText('deployed')).toBeDefined()
    const legacyRow = screen.getByText('legacy').closest('tr') as HTMLElement
    // No params key → the Params cell falls back to an em dash like the other empty cells.
    expect(within(legacyRow).getAllByText('–').length).toBeGreaterThan(0)
    expect(within(legacyRow).queryByText(/max_iter/)).toBeNull()
  })
})
