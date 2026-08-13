/**
 * A debounced function: trailing calls fire once `ms` after the last call.
 * `flush()` runs a pending call immediately (used before form saves so no
 * keystroke is lost); `cancel()` drops it (unmount cleanup).
 */
export interface Debounced<A extends unknown[]> {
  (...args: A): void
  flush: () => void
  cancel: () => void
}

export function debounce<A extends unknown[]>(fn: (...args: A) => void, ms = 300): Debounced<A> {
  let timer: ReturnType<typeof setTimeout> | null = null
  let lastArgs: A | null = null

  const debounced = (...args: A) => {
    lastArgs = args
    if (timer !== null) clearTimeout(timer)
    timer = setTimeout(() => {
      timer = null
      const call = lastArgs
      lastArgs = null
      if (call !== null) fn(...call)
    }, ms)
  }

  debounced.flush = () => {
    if (timer !== null) {
      clearTimeout(timer)
      timer = null
    }
    const call = lastArgs
    lastArgs = null
    if (call !== null) fn(...call)
  }

  debounced.cancel = () => {
    if (timer !== null) clearTimeout(timer)
    timer = null
    lastArgs = null
  }

  return debounced
}
