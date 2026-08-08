import type { ReactNode } from 'react'

export type BadgeVariant = 'ready' | 'missing' | 'warn' | 'info'

export function Badge({ variant, children }: { variant: BadgeVariant; children: ReactNode }) {
  return <span className={`badge ${variant}`}>{children}</span>
}
