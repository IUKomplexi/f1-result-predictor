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
  it('lists positions on the left, seeded from the prediction grid', () => {
    render(<GridEditor drivers={drivers} values={null} onChange={() => {}} onReset={() => {}} />)
    const cells = screen.getAllByRole('cell')
    // Position column first, driver dropdown second.
    expect(cells[0].textContent).toBe('1')
    expect(cells[2].textContent).toBe('2')
    const p1 = screen.getByLabelText('Driver at position 1') as HTMLSelectElement
    const p2 = screen.getByLabelText('Driver at position 2') as HTMLSelectElement
    expect(p1.value).toBe('russell') // model grid: russell P1
    expect(p2.value).toBe('norris') // model grid: norris P2
  })

  it('assigning a driver to a position reports the override', () => {
    const onChange = vi.fn()
    render(<GridEditor drivers={drivers} values={null} onChange={onChange} onReset={() => {}} />)
    const p2 = screen.getByLabelText('Driver at position 2') as HTMLSelectElement
    fireEvent.change(p2, { target: { value: 'russell' } })
    expect(onChange).toHaveBeenCalledWith({ russell: '2' })
  })

  it('does not swap a driver when assigning an occupied driver elsewhere', () => {
    const onChange = vi.fn()
    render(<GridEditor drivers={drivers} values={{ russell: '2' }} onChange={onChange} onReset={() => {}} />)
    const p1 = screen.getByLabelText('Driver at position 1') as HTMLSelectElement
    fireEvent.change(p1, { target: { value: 'russell' } })
    expect(onChange).not.toHaveBeenCalled()
  })

  it('disables an explicitly selected driver in other positions', () => {
    render(
      <GridEditor
        drivers={drivers}
        values={{ russell: '1' }}
        onChange={() => {}}
        onReset={() => {}}
      />,
    )
    const p2 = screen.getByLabelText('Driver at position 2') as HTMLSelectElement
    const russell = Array.from(p2.options).find((option) => option.value === 'russell')
    expect(russell).toBeUndefined()
    expect(p2.options[0].textContent).toBe('Use model grid')
  })

  it('"Use model grid" clears the position override', () => {
    const onChange = vi.fn()
    render(<GridEditor drivers={drivers} values={{ norris: '1' }} onChange={onChange} onReset={() => {}} />)
    const p1 = screen.getByLabelText('Driver at position 1') as HTMLSelectElement
    expect(p1.value).toBe('norris')
    fireEvent.change(p1, { target: { value: '' } })
    expect(onChange).toHaveBeenCalledWith({})
  })

  it('reset is available only once the grid is dirty', () => {
    const onReset = vi.fn()
    const { rerender } = render(
      <GridEditor drivers={drivers} values={null} onChange={() => {}} onReset={onReset} />,
    )
    const reset = () => screen.getByText('Reset to model grid') as HTMLButtonElement
    expect(reset().disabled).toBe(true)
    rerender(<GridEditor drivers={drivers} values={{ russell: '1' }} onChange={() => {}} onReset={onReset} />)
    expect(reset().disabled).toBe(false)
    fireEvent.click(reset())
    expect(onReset).toHaveBeenCalledOnce()
  })
})
