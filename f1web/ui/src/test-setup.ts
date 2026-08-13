// Test-environment setup shared by every vitest suite (see vitest.config.ts).
//
// 1. Load preact/compat: the preset aliases `react` -> `preact/compat` and the
//    app components rely on the compat diff hook (it maps onChange -> input
//    events on form elements, exactly like the built dashboard). Importing it
//    once here activates that hook for all tests.
// 2. Stub ResizeObserver: jsdom does not implement it, but ui/Chart.tsx
//    measures its wrapper with one. The stub reports a fixed width on
//    observe() so charts render instead of waiting for a layout that never
//    happens in jsdom.
import 'preact/compat'

class ResizeObserverStub {
  private callback: ResizeObserverCallback

  constructor(callback: ResizeObserverCallback) {
    this.callback = callback
  }

  observe(target: Element): void {
    const contentRect = { width: 640, height: 220 } as DOMRectReadOnly
    this.callback(
      [{ target, contentRect } as ResizeObserverEntry],
      this as unknown as ResizeObserver,
    )
  }

  unobserve(): void {}
  disconnect(): void {}
}

if (typeof globalThis.ResizeObserver === 'undefined') {
  ;(globalThis as unknown as Record<string, unknown>).ResizeObserver = ResizeObserverStub
}
