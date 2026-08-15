import { fireEvent, render, screen } from '@testing-library/preact'
import { describe, expect, it, vi } from 'vitest'
import { ModelParams } from './ModelParams'

const KEYS = ['max_iter', 'learning_rate', 'max_depth', 'l2_regularization', 'min_samples_leaf']

describe('ModelParams', () => {
  it('renders a number field per key with the current values', () => {
    render(
      <ModelParams
        keys={KEYS}
        value={{ max_iter: 200, learning_rate: 0.05, l2_regularization: 10 }}
        onChange={() => {}}
      />,
    )
    expect(screen.getByLabelText('max_iter')).toBeDefined()
    expect(screen.getByLabelText('min_samples_leaf')).toBeDefined()
    const maxIter = screen.getByLabelText('max_iter') as HTMLInputElement
    expect(maxIter.value).toBe('200')
    expect((screen.getByLabelText('learning_rate') as HTMLInputElement).value).toBe('0.05')
    // 10.0 is a JS number -> renders as 10.
    expect((screen.getByLabelText('l2_regularization') as HTMLInputElement).value).toBe('10')
    // Absent keys render empty.
    expect((screen.getByLabelText('max_depth') as HTMLInputElement).value).toBe('')
  })

  it('reports edits as the full params dict', () => {
    const onChange = vi.fn()
    render(<ModelParams keys={KEYS} value={{ max_iter: 200 }} onChange={onChange} />)
    // fireEvent.input: compat maps onChange -> input events; fireEvent.change
    // would dispatch a change event no listener is attached for.
    fireEvent.input(screen.getByLabelText('learning_rate'), { target: { value: '0.1' } })
    expect(onChange).toHaveBeenCalledWith({ max_iter: 200, learning_rate: 0.1 })
  })

  it('locks the inputs when disabled and renders the hint', () => {
    render(
      <ModelParams
        keys={KEYS}
        value={{ max_iter: 200 }}
        disabled
        hint="Locked: set per training on the Train tab."
      />,
    )
    const maxIter = screen.getByLabelText('max_iter') as HTMLInputElement
    expect(maxIter.disabled).toBe(true)
    expect(screen.getByText(/set per training on the Train tab/)).toBeDefined()
  })
})
