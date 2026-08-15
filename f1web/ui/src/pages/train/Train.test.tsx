import { fireEvent, render, screen, waitFor } from '@testing-library/preact'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getConfig, getModels, getStatus, postJob } from '../../api/client'
import { Train } from './Train'

vi.mock('../../api/client', () => ({
  getConfig: vi.fn(),
  getModels: vi.fn(),
  getStatus: vi.fn(),
  postJob: vi.fn(),
}))

beforeEach(() => {
  vi.clearAllMocks()
})

const status = {
  seasons: { start: 2014, end: 2026, data_start: 2014, data_end: 2026 },
  model: { has_checkpoint: true, has_calibrators: true },
  data: { has_dataset: true, has_raw_cache: true },
  reports: { has_backtest: true, has_calibration: true },
  dashboard: { built: true },
}

const DEPLOYED_PARAMS = {
  max_iter: 200,
  learning_rate: 0.05,
  max_depth: 3,
  l2_regularization: 10,
  min_samples_leaf: 50,
}

const config = {
  config: { model: { params: { max_iter: 200, learning_rate: 0.05 } } },
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
  model_params_keys: Object.keys(DEPLOYED_PARAMS),
  jobs: [],
}

const models = {
  models: {
    hurdle: { checkpoint: 'data/model/hurdle.joblib', params: DEPLOYED_PARAMS },
  },
  default: 'data/model/hurdle.joblib',
}

describe('Train', () => {
  it('accepts numeric model names (digits are allowed)', async () => {
    vi.mocked(getStatus).mockResolvedValue(status as never)
    vi.mocked(getConfig).mockResolvedValue(config as never)
    vi.mocked(getModels).mockResolvedValue(models as never)
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
    vi.mocked(getModels).mockResolvedValue(models as never)
    render(<Train />)
    const input = (await screen.findByLabelText('Model name')) as HTMLInputElement
    fireEvent.change(input, { target: { value: 'hurdle 2026!' } })
    expect(input.value).toBe('hurdle2026')
  })

  it('prefills the hyperparameter editor from the deployed model', async () => {
    vi.mocked(getStatus).mockResolvedValue(status as never)
    vi.mocked(getConfig).mockResolvedValue(config as never)
    vi.mocked(getModels).mockResolvedValue(models as never)
    render(<Train />)
    expect((await screen.findByLabelText('max_iter')) as HTMLInputElement).toBeDefined()
    // Wait for the prefill (deployed params win over the config fallback).
    expect((await screen.findByDisplayValue('200')) as HTMLInputElement).toBeDefined()
    expect((screen.getByLabelText('learning_rate') as HTMLInputElement).value).toBe('0.05')
    expect((screen.getByLabelText('min_samples_leaf') as HTMLInputElement).value).toBe('50')
  })

  it('sends the hyperparameters with the train job payload', async () => {
    vi.mocked(getStatus).mockResolvedValue(status as never)
    vi.mocked(getConfig).mockResolvedValue(config as never)
    vi.mocked(getModels).mockResolvedValue(models as never)
    vi.mocked(postJob).mockResolvedValue({ id: 'job-1' } as never)
    render(<Train />)
    // Editor is prefilled before the run is submitted.
    await screen.findByDisplayValue('200')
    fireEvent.click(screen.getByRole('button', { name: 'Train model' }))
    await waitFor(() => expect(postJob).toHaveBeenCalledTimes(1))
    expect(postJob).toHaveBeenCalledWith(
      'train',
      expect.objectContaining({ params: DEPLOYED_PARAMS }),
    )
  })

  it('sends edited hyperparameters with the train job payload', async () => {
    vi.mocked(getStatus).mockResolvedValue(status as never)
    vi.mocked(getConfig).mockResolvedValue(config as never)
    vi.mocked(getModels).mockResolvedValue(models as never)
    vi.mocked(postJob).mockResolvedValue({ id: 'job-2' } as never)
    render(<Train />)
    // Wait for the prefill to settle first so the edit cannot race it.
    await screen.findByDisplayValue('200')
    const maxIter = (await screen.findByLabelText('max_iter')) as HTMLInputElement
    // fireEvent.input: compat maps onChange -> input events on form elements.
    fireEvent.input(maxIter, { target: { value: '500' } })
    // Wait for the edit to flush into component state before submitting, so
    // buildPayload closes over the updated params.
    await waitFor(() => expect((screen.getByLabelText('max_iter') as HTMLInputElement).value).toBe('500'))
    fireEvent.click(screen.getByRole('button', { name: 'Train model' }))
    await waitFor(() => expect(postJob).toHaveBeenCalledTimes(1))
    expect(postJob).toHaveBeenCalledWith(
      'train',
      expect.objectContaining({ params: expect.objectContaining({ max_iter: 500 }) }),
    )
  })
})
