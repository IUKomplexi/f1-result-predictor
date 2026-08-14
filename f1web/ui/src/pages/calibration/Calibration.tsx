import { getCalibration, type Calibration, type CalibrationTarget } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { fmtNumber } from '../../lib/format'
import { Badge } from '../../components/ui/Badge'
import { Chart, type ChartDatum, type ChartSeries } from '../../components/ui/Chart'
import { ErrorState, Skeleton } from '../../components/ui/DataState'

const TARGET_LABEL: Record<string, string> = {
  scored: 'P scored',
  top3: 'P top 3',
  win: 'P win',
}

/**
 * Calibration report (reports/calibration.json): per-target raw vs calibrated
 * Brier plus reliability curves for the deployed model. Calibration itself
 * runs automatically as part of every Train job — this tab is the
 * diagnostics view only, so there is no run trigger here.
 */
export function Calibration() {
  const { state, retry } = useApi('calibration', () => getCalibration())

  return (
    <>
      <section className="card">
        <h2 className="card-title">Calibration</h2>
        <p className="context-note">
          Calibration runs automatically with every Train job; this tab shows
          the report for the deployed model.
        </p>
      </section>
      {state.phase === 'loading' ? (
        <Skeleton rows={6} />
      ) : state.phase === 'error' ? (
        <ErrorState message={state.message} onRetry={retry} />
      ) : (
        <CalibrationView calibration={state.data} />
      )}
    </>
  )
}

/** The reports/calibration.json snapshot: Brier table + reliability curves. */
export function CalibrationView({ calibration }: { calibration: Calibration }) {
  const entries = Object.entries(calibration.targets)
  return (
    <>
      <section className="card">
        <h2 className="card-title">Brier score — raw vs calibrated</h2>
        <p className="context-note">
          {calibration.context || 'How calibration was tested.'}
        </p>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Target</th>
                <th scope="col" className="num">Deployed</th>
                <th scope="col" className="num">Raw</th>
                <th scope="col" className="num">Calibrated</th>
                <th scope="col" className="num">Δ</th>
              </tr>
            </thead>
            <tbody>
              {entries.map(([target, value]) => (
                <tr key={target}>
                  <td>{TARGET_LABEL[target] ?? target}</td>
                  <td className="num">
                    {value.deployed ? (
                      <Badge variant="ready">Yes</Badge>
                    ) : (
                      <Badge variant="missing">No</Badge>
                    )}
                  </td>
                  <td className="num">{fmtNumber(value.brier_raw, 4)}</td>
                  <td className="num">{fmtNumber(value.brier_calibrated, 4)}</td>
                  <td className="num">
                    <span className={value.delta <= 0 ? 'delta-good' : 'delta-bad'}>
                      {fmtNumber(value.delta, 4)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <h2 className="card-title">Reliability curves</h2>
        <p className="context-note">
          Predicted probability (x) vs observed frequency (y). The dashed line is
          perfect calibration.
        </p>
        <div className="chart-grid">
          {entries.map(([target, value]) => (
            <figure key={target} className="chart-figure">
              <figcaption>{TARGET_LABEL[target] ?? target}</figcaption>
              <ReliabilityChart target={value} />
            </figure>
          ))}
        </div>
      </section>
    </>
  )
}

function ReliabilityChart({ target }: { target: CalibrationTarget }) {
  const data: ChartDatum[] = target.reliability.map((bin) => ({
    predicted: bin.mean_pred,
    observed: bin.observed,
  }))
  const series: ChartSeries[] = [
    { key: 'observed', name: 'Observed', color: '#e10600', strokeWidth: 2, dot: true },
  ]
  return (
    <Chart
      data={data}
      xKey="predicted"
      xType="number"
      xDomain={[0, 1]}
      yDomain={[0, 1]}
      series={series}
      referenceLine={{ from: [0, 0], to: [1, 1] }}
      valueFormat={(v) => fmtNumber(v, 4)}
    />
  )
}
