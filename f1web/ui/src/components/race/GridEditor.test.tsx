import { fireEvent, render, screen } from '@testing-library/preact'
import { describe, expect, it, vi } from 'vitest'
import type { PredictionRow } from '../../api/client'
import { GridEditor } from './GridEditor'

const drivers: PredictionRow[] = [
  {
    pred_rank: 1,
    driver_id: 'russell',
    constructor_id: 'mercedes',
    grid: 1,
    expected_points: 15,
    p_scored: 0.9,
    p_top3: 0.6,
    p_win: 0.5,
    actual_points: null,
    actual_position: null,
  },
  {
    pred_rank: 2,
    driver_id: 'norris',
    constructor_id: 'mclaren',
    grid: 2,
    expected_points: 12,
    p_scored: 0.8,
    p_top3: 0.5,
    p_win: 0.3,
    actual_points: null,
    actual_position: null,
  },
]

describe('GridEditor', () => {
  it('seeds dropdowns from the prediction grid and reports edits', () => {
    const onChange = vi.fn()
    render(<GridEditor drivers={drivers} values={null} onChange={onChange} onReset={() => {}} />)
    const select = screen.getByLabelText('Grid position for Russell') as HTMLSelectElement
    expect(select.value).toBe('1')
    fireEvent.change(select, { target: { value: '2' } })
    expect(onChange).toHaveBeenCalledWith({ russell: '2' })
  })

  it('offers a "Use model grid" no-override option plus one option per position', () => {
    render(<GridEditor drivers={drivers} values={null} onChange={() => {}} onReset={() => {}} />)
    const select = screen.getByLabelText('Grid position for Russell') as HTMLSelectElement
    const options = Array.from(select.options).map((o) => o.value)
    expect(options[0]).toBe('') // "Use model grid"
    expect(options).toContain('1')
    expect(options).toContain('2')
  })

  it('shows edited values over the seed and accumulates edits per driver', () => {
    const onChange = vi.fn()
    render(<GridEditor drivers={drivers} values={{ norris: '1' }} onChange={onChange} onReset={() => {}} />)
    const norris = screen.getByLabelText('Grid position for Norris') as HTMLSelectElement
    expect(norris.value).toBe('1')
    fireEvent.change(norris, { target: { value: '2' } })
    expect(onChange).toHaveBeenCalledWith({ norris: '2' })
  })

  it('reset is available only once the grid is dirty', () => {
    const onReset = vi.fn()
    const { rerender } = render(
      <GridEditor drivers={drivers} values={null} onChange={() => {}} onReset={onReset} />,
    )
    const reset = () => screen.getByText('Reset to model grid') as HTMLButtonElement
    expect(reset().disabled).toBe(true)
    rerender(<GridEditor drivers={drivers} values={{ russell: '3' }} onChange={() => {}} onReset={onReset} />)
    expect(reset().disabled).toBe(false)
    fireEvent.click(reset())
    expect(onReset).toHaveBeenCalledOnce()
  })
})
