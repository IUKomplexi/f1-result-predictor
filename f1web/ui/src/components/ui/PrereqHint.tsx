import type { ReactNode } from 'react'

/**
 * Inline prerequisite hint for pipeline tabs: rendered only while the
 * upstream artifact is missing, so a fresh checkout explains what to run
 * next instead of silently producing empty results.
 */
export function PrereqHint({ when, children }: { when: boolean; children: ReactNode }) {
  if (!when) return null
  return <p className="prereq-hint">{children}</p>
}
