import { fireEvent, render, screen, waitFor } from '@testing-library/preact'
import { describe, expect, it, vi } from 'vitest'
import { getConfig } from '../../api/client'
import { FeatureToggles, NO_FEATURE_OVERRIDES, type FeatureOverride } from './FeatureToggles'

vi.mock('../../api/client', () => ({
  getConfig: vi.fn(),
}))

const config = {
  config: {},
  schema: [],
  features: {
    registry: ['grid', 'season', 'driver_id'],
    defaults: ['grid', 'driver_id'],
    categories: { grid: 'core', season: 'selectable', driver_id: 'core' },
    category_meta: [
      { id: 'core', label: 'Core — on by default' },
      { id: 'selectable', label: 'Selectable — off by default' },
    ],
  },
  seasons: { min: 2014, max: 2026, data_start: 2014, data_end: 2026 },
  model_params_keys: [],
  jobs: [],
}

describe('FeatureToggles', () => {
  it('shows grouped checkboxes with the effective default state', async () => {
    vi.mocked(getConfig).mockResolvedValue(config as never)
    render(<FeatureToggles value={NO_FEATURE_OVERRIDES} onChange={() => {}} />)
    await waitFor(() => expect(screen.getByLabelText('grid feature')).toBeDefined())
    const grid = screen.getByLabelText('grid feature') as HTMLInputElement
    const season = screen.getByLabelText('season feature') as HTMLInputElement
    expect(grid.checked).toBe(true) // core -> on by default
    expect(season.checked).toBe(false) // selectable -> off by default
    expect(screen.getByText('Core — on by default')).toBeDefined()
    expect(screen.getByText('Selectable — off by default')).toBeDefined()
  })

  it('turns a default-on feature into a disable override when unchecked', async () => {
    vi.mocked(getConfig).mockResolvedValue(config as never)
    const onChange = vi.fn()
    render(<FeatureToggles value={NO_FEATURE_OVERRIDES} onChange={onChange} />)
    await waitFor(() => expect(screen.getByLabelText('grid feature')).toBeDefined())
    fireEvent.click(screen.getByLabelText('grid feature'))
    expect(onChange).toHaveBeenCalledWith({ enable: [], disable: ['grid'] })
  })

  it('turns a default-off feature into an enable override when checked', async () => {
    vi.mocked(getConfig).mockResolvedValue(config as never)
    const onChange = vi.fn()
    render(<FeatureToggles value={NO_FEATURE_OVERRIDES} onChange={onChange} />)
    await waitFor(() => expect(screen.getByLabelText('season feature')).toBeDefined())
    fireEvent.click(screen.getByLabelText('season feature'))
    expect(onChange).toHaveBeenCalledWith({ enable: ['season'], disable: [] })
  })

  it('drops the override when the checkbox returns to the default state', async () => {
    vi.mocked(getConfig).mockResolvedValue(config as never)
    const onChange = vi.fn()
    const value: FeatureOverride = { enable: [], disable: ['grid'] }
    render(<FeatureToggles value={value} onChange={onChange} />)
    await waitFor(() => expect(screen.getByLabelText('grid feature')).toBeDefined())
    const grid = screen.getByLabelText('grid feature') as HTMLInputElement
    expect(grid.checked).toBe(false) // forced off
    fireEvent.click(grid) // back to default (on)
    expect(onChange).toHaveBeenCalledWith({ enable: [], disable: [] })
  })

  it('reset restores no overrides', async () => {
    vi.mocked(getConfig).mockResolvedValue(config as never)
    const onChange = vi.fn()
    const value: FeatureOverride = { enable: ['season'], disable: ['grid'] }
    render(<FeatureToggles value={value} onChange={onChange} />)
    await waitFor(() => expect(screen.getByText('Reset to config defaults')).toBeDefined())
    fireEvent.click(screen.getByText('Reset to config defaults'))
    expect(onChange).toHaveBeenCalledWith(NO_FEATURE_OVERRIDES)
  })
})
