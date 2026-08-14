import { render, screen } from '@testing-library/preact'
import { describe, expect, it } from 'vitest'
import { BacktestRunResult } from './BacktestRunResult'

/** Realistic multi-model job result (shape as produced by model/evaluate.py). */
const multiModelJob = {
  log: ['Scoring with checkpoint data/model/hurdle.joblib'],
  result: {
    checkpoint: 'data/model/hurdle.joblib',
    models: ['alpha', 'beta'],
    overall: {
      model: { winner_hit: 0.55, top3_overlap: 0.6, top10_overlap: 0.7, spearman: 0.6, mae: 3.1 },
      grid: { winner_hit: 0.53, top3_overlap: 0.69, top10_overlap: 0.76, spearman: 0.62, mae: 2.8 },
      championship: { winner_hit: 0.46, top3_overlap: 0.6, top10_overlap: 0.74, spearman: 0.61, mae: 4.1 },
      zero: { winner_hit: 0.53, top3_overlap: 0.69, top10_overlap: 0.76, spearman: 0.62, mae: 5.0 },
    },
    snapshot: {
      models: {
        alpha: {
          overall: {
            model: { winner_hit: 0.55, top3_overlap: 0.6, top10_overlap: 0.7, spearman: 0.6, mae: 3.1 },
          },
        },
        beta: {
          overall: {
            model: { winner_hit: 0.6, top3_overlap: 0.65, top10_overlap: 0.72, spearman: 0.63, mae: 2.9 },
          },
        },
      },
    },
  },
}

describe('BacktestRunResult', () => {
  it('renders one row per compared model, labeled with the model stem', () => {
    render(<BacktestRunResult job={multiModelJob} />)
    const rows = [...screen.getAllByRole('row')].map((row) => row.textContent ?? '')
    const modelRow = rows.find((r) => r.includes('alpha') && r.includes('0.55'))
    const betaRow = rows.find((r) => r.includes('beta') && r.includes('0.6'))
    expect(modelRow).toBeDefined()
    expect(betaRow).toBeDefined()
    // Each model appears in its own row with its own metric values.
    expect(modelRow).not.toContain('beta')
    expect(betaRow).not.toContain('alpha')
  })

  it('titles the baselines table with the first model name', () => {
    render(<BacktestRunResult job={multiModelJob} />)
    expect(screen.getByText('hurdle vs baselines (mean)')).toBeDefined()
  })

  it('renders the baselines table even for a single-model run', () => {
    render(
      <BacktestRunResult
        job={{
          log: [],
          result: {
            checkpoint: 'data/model/hurdle.joblib',
            overall: {
              model: { winner_hit: 0.55, top3_overlap: 0.6, top10_overlap: 0.7, spearman: 0.6, mae: 3.1 },
            },
          },
        }}
      />,
    )
    expect(screen.getByText('hurdle vs baselines (mean)')).toBeDefined()
    // No compared-models section for a single model.
    expect(screen.queryByText('Compared models (mean)')).toBeNull()
  })
})
