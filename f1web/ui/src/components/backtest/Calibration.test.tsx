import { render, screen } from '@testing-library/preact'
import { describe, expect, it } from 'vitest'
import type { Calibration } from '../../api/client'
import { CalibrationView } from './Calibration'

const snapshot: Calibration = {
  context: 'fit on 2023-2024, evaluated on 2025',
  deployed: ['scored'],
  targets: {
    scored: {
      brier_raw: 0.12,
      brier_calibrated: 0.1,
      delta: -0.02,
      deployed: true,
      reliability: [
        { mean_pred: 0.1, observed: 0.08, n: 50 },
        { mean_pred: 0.9, observed: 0.85, n: 40 },
      ],
    },
    win: {
      brier_raw: 0.05,
      brier_calibrated: 0.052,
      delta: 0.002,
      deployed: false,
      reliability: [{ mean_pred: 0.05, observed: 0.04, n: 60 }],
    },
  },
}

describe('CalibrationView', () => {
  it('renders the Brier table with deployment badges and the context note', () => {
    render(<CalibrationView calibration={snapshot} />)
    expect(screen.getByText('Brier score — raw vs calibrated')).toBeTruthy()
    expect(screen.getByText('fit on 2023-2024, evaluated on 2025')).toBeTruthy()
    // Each target appears twice: as a Brier-table row and as a chart caption.
    expect(screen.getAllByText('P scored').length).toBe(2)
    expect(screen.getAllByText('P win').length).toBe(2)
    // One deployed (scored), one not (win).
    expect(screen.getAllByText('Yes').length).toBe(1)
    expect(screen.getAllByText('No').length).toBe(1)
  })

  it('renders one reliability chart per target', () => {
    const { container } = render(<CalibrationView calibration={snapshot} />)
    const charts = container.querySelectorAll('svg[role="img"]')
    expect(charts.length).toBe(2)
  })
})
