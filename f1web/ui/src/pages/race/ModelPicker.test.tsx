import { fireEvent, render, screen } from '@testing-library/preact'
import { describe, expect, it, vi } from 'vitest'
import type { ModelsResponse } from '../../api/client'
import { ModelPicker } from './ModelPicker'

const models: ModelsResponse = {
  default: 'data/model/hurdle.joblib',
  models: {
    alpha: { checkpoint: 'data/model/alpha.joblib' },
    beta: { checkpoint: 'data/model/beta.joblib' },
  },
}

describe('ModelPicker', () => {
  it('lists saved models plus the config default entry', () => {
    render(<ModelPicker models={models} value="alpha" onChange={() => {}} />)
    const options = screen.getAllByRole('option').map((o) => (o as HTMLOptionElement).value)
    expect(options).toContain('default')
    expect(options).toContain('alpha')
    expect(options).toContain('beta')
  })

  it('marks the deployed checkpoint among the saved models', () => {
    const deployed: ModelsResponse = {
      default: 'data/model/alpha.joblib',
      models: { alpha: { checkpoint: 'data/model/alpha.joblib' } },
    }
    render(<ModelPicker models={deployed} value="alpha" onChange={() => {}} />)
    const labels = screen.getAllByRole('option').map((o) => (o as HTMLOptionElement).textContent)
    expect(labels).toContain('alpha (deployed)')
  })

  it('reports choice changes', () => {
    const onChange = vi.fn()
    render(<ModelPicker models={models} value="default" onChange={onChange} />)
    const select = screen.getByLabelText('Model') as HTMLSelectElement
    fireEvent.change(select, { target: { value: 'alpha' } })
    expect(onChange).toHaveBeenCalledWith('alpha')
  })

  it('shows a disabled loading select before models load', () => {
    render(<ModelPicker models={null} value="default" onChange={() => {}} />)
    expect((screen.getByLabelText('Model') as HTMLSelectElement).disabled).toBe(true)
  })
})
