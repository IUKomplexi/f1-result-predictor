import { Component, Fragment, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
}

interface State {
  error: Error | null
  resetKey: number
}

/**
 * Catches render-time errors from any tab so a single crash can't blank the
 * whole SPA (before this, one throw in a lazy tab unmounted the entire tree).
 * Errors are surfaced inline with the message and a retry that remounts the
 * child subtree.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, resetKey: 0 }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // The real message is the thing to fix; log it so it can be captured.
    console.error('Dashboard render error:', error, info.componentStack)
  }

  private retry = () => {
    this.setState((s) => ({ error: null, resetKey: s.resetKey + 1 }))
  }

  render() {
    if (this.state.error) {
      return (
        <section className="state-block" role="alert">
          <h2>Something went wrong in this view</h2>
          <p className="save-status error">{this.state.error.message}</p>
          <button type="button" className="button" onClick={this.retry}>
            Try again
          </button>
        </section>
      )
    }
    return <Fragment key={this.state.resetKey}>{this.props.children}</Fragment>
  }
}
