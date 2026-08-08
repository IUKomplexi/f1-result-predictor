import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { getCalibration, type Calibration, type CalibrationTarget } from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { fmtNumber } from '../../lib/format'
import { Badge } from '../ui/Badge'
import { ErrorState, Skeleton } from '../ui/DataState'
import './Calibration.css'

const TARGET_LABEL: Record<string, string> = {
  scored: 'P scored',
  top3: 'P top 3',
  win: 'P win',
}

export function Calibration() {
  const { state, retry } = useApi('calibration', () => getCalibration())
  if (state.phase === 'loading') return <Skeleton rows={6} />
  if (state.phase === 'error') return <ErrorState message={state.message} onRetry={retry} />
  return <CalibrationView calibration={state.data} />
}

function CalibrationView({ calibration }: { calibration: Calibration }) {
  const entries = Object.entries(calibration.targets)
  return (
    <>
      <section className="card">
        <h2 className="card-title">Brier score — raw vs calibrated</h2>
        <p className="context-note">{calibration.context || 'Calibration evaluation (out-of-sample).'}</p>
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
  const data = target.reliability.map((bin) => ({
    predicted: bin.mean_pred,
    observed: bin.observed,
  }))
  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
          <CartesianGrid stroke="#2e2e3a" strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="predicted"
            domain={[0, 1]}
            tick={{ fill: '#a8a8b5', fontSize: 11 }}
            stroke="#2e2e3a"
          />
          <YAxis
            type="number"
            domain={[0, 1]}
            tick={{ fill: '#a8a8b5', fontSize: 11 }}
            stroke="#2e2e3a"
          />
          <Tooltip
            contentStyle={{
              background: '#23232e',
              border: '1px solid #2e2e3a',
              borderRadius: 8,
              color: '#f2f2f5',
            }}
          />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke="#6f6f7d"
            strokeDasharray="4 4"
          />
          <Line
            type="linear"
            dataKey="observed"
            name="Observed"
            stroke="#e10600"
            strokeWidth={2}
            dot={{ r: 3, fill: '#e10600' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
