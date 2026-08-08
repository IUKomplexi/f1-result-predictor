import { useEffect, useRef, useState } from 'react'
import { ApiError } from '../api/client'

export type LoadState<T> =
  | { phase: 'loading' }
  | { phase: 'error'; message: string }
  | { phase: 'ready'; data: T }

export interface UseApiResult<T> {
  state: LoadState<T>
  retry: () => void
}

/**
 * Minimal async-state hook: loading / error (with retry) / ready.
 *
 * The effect re-runs whenever `key` changes or `retry()` is called. Callers
 * pass a stable key derived from the request parameters (e.g. `season-2024`),
 * and a loader that captures those parameters.
 */
export function useApi<T>(key: string, loader: () => Promise<T>): UseApiResult<T> {
  const [attempt, setAttempt] = useState(0)
  const [state, setState] = useState<LoadState<T>>({ phase: 'loading' })
  const loaderRef = useRef(loader)
  loaderRef.current = loader

  useEffect(() => {
    let cancelled = false
    setState({ phase: 'loading' })
    loaderRef
      .current()
      .then((data) => {
        if (!cancelled) setState({ phase: 'ready', data })
      })
      .catch((error: unknown) => {
        if (cancelled) return
        const message = error instanceof ApiError ? error.message : String(error)
        setState({ phase: 'error', message })
      })
    return () => {
      cancelled = true
    }
  }, [key, attempt])

  return { state, retry: () => setAttempt((n) => n + 1) }
}
