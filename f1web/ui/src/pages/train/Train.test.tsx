import { fireEvent, render, screen } from '@testing-library/preact'
import { describe, expect, it, vi } from 'vitest'
import { getConfig, getStatus } from '../../api/client'
import { Train } from './Train'

vi.mock('../../api/client', () => ({
  getConfig: vi.fn(),
  getStatus: vi.fn(),
}))

const status = {
  seasons: { start: 2014, end: 2026, data_start: 2014, data_end: 2026 },
  model: { has_checkpoint: true, has_calibrators: true },
  data: { has_dataset: true, has_raw_cache: true },
  reports: { has_backtest: true, has_calibration: true },
  dashboard: { built: true },
}

const config = {
  config: {},
  schema: [],
  features: {
    registry: ['grid', 'season'],
    defaults: ['grid'],
    categories: { grid: 'core', season: 'selectable' },
    category_meta: [
      { id: 'core', label: 'Core — on by default' },
      { id: 'selectable', label: 'Selectable — off by default' },
    ],
  },
  seasons: { min: 2014, max: 2026, data_start: 2014, data_end: 2026 },
  model_params_keys: [],
  jobs: [],
}

describe('Train', () => {
  it('accepts numeric model names (digits are allowed)', async () => {
    vi.mocked(getStatus).mockResolvedValue(status as never)
    vi.mocked(getConfig).mockResolvedValue(config as never)
    render(<Train />)
    const input = (await screen.findByLabelText('Model name')) as HTMLInputElement
    fireEvent.change(input, { target: { value: '2018' } })
    expect(input.value).toBe('2018')
    // No validation error is shown for a numeric name.
    expect(screen.queryByText(/cannot be a number/i)).toBeNull()
  })

  it('sanitizes characters outside the allowed name set', async () => {
    vi.mocked(getStatus).mockResolvedValue(status as never)
    vi.mocked(getConfig).mockResolvedValue(config as never)
    render(<Train />)
    const input = (await screen.findByLabelText('Model name')) as HTMLInputElement
    fireEvent.change(input, { target: { value: 'hurdle 2026!' } })
    expect(input.value).toBe('hurdle2026')
  })
})
