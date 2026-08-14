/**
 * The cross-tab contract shared between the app shell (App.tsx) and every tab
 * page. Tabs receive these props and use onNavigate to jump between tabs, so
 * the types live here instead of inside App.tsx — importing them from a tab
 * must not pull in the whole shell.
 */

export type TabId = 'status' | 'race' | 'history' | 'data' | 'train' | 'backtest' | 'settings'

/** Cross-tab navigation payload (e.g. Race History → a specific race). */
export interface NavState {
  season?: number
  round?: number
}

export interface TabProps {
  onNavigate?: (tabId: string, state?: NavState) => void
  navState?: NavState | null
  /** A tab can veto navigation (e.g. Settings with unsaved edits). The guard
   *  returns false to block; the component retries via onNavigate itself. */
  setNavigateGuard?: (
    guard: ((tabId: string, state?: NavState) => boolean) | null,
  ) => void
}
