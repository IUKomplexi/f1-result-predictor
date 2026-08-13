import { useEffect, useMemo, useRef, useState, type MouseEvent, type TouchEvent } from 'react'
import './Chart.css'

/**
 * Lightweight SVG line chart replacing recharts. Handles the two chart shapes
 * the dashboard needs: a category/x-number multi-series line chart (backtest
 * trends) and a 0–1 number-x reliability curve with a diagonal reference line
 * (calibration). Ships its own responsive width, axis ticks, legend, reference
 * line and hover tooltip so the d3/recharts dependency tree stays out.
 */

export interface ChartSeries {
  key: string
  name: string
  color: string
  strokeWidth?: number
  dot?: boolean
  /** When false (default) a missing value breaks the line into a gap. */
  connectNulls?: boolean
}

export interface ChartDatum {
  [key: string]: string | number | undefined
}

export interface ReferenceLineSpec {
  /** Horizontal dashed line at this y value. */
  y?: number
  /** Diagonal dashed line from `from` to `to` (both `[x, y]`). */
  from?: [number, number]
  to?: [number, number]
  label?: string
}

interface ChartProps {
  data: ChartDatum[]
  /** Row key holding the x value. */
  xKey: string
  xType?: 'category' | 'number'
  xDomain?: [number, number]
  yDomain?: [number, number]
  series: ChartSeries[]
  height?: number
  referenceLine?: ReferenceLineSpec
  valueFormat?: (value: number) => string
  /** Accessible description of what the chart plots (SVG role="img"). */
  ariaLabel?: string
}

const PAD = { top: 10, right: 14, bottom: 26, left: 44 }

export function Chart({
  data,
  xKey,
  xType = 'category',
  xDomain,
  yDomain,
  series,
  height = 220,
  referenceLine,
  valueFormat = (v) => v.toFixed(3),
  ariaLabel,
}: ChartProps) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const [width, setWidth] = useState(0)
  const [hover, setHover] = useState<number | null>(null)

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width ?? 0
      setWidth((prev) => (prev === w ? prev : w))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const { plot, xScale, yScale, yTicks, xTicks, lines } = useMemo(() => {
    const plotWidth = Math.max(0, width - PAD.left - PAD.right)
    const plotHeight = Math.max(0, height - PAD.top - PAD.bottom)

    // x positions.
    const numX = xType === 'number'
    const xs = data.map((row) => Number(row[xKey]))
    const xMin = numX ? (xDomain?.[0] ?? Math.min(...xs)) : 0
    const xMax = numX ? (xDomain?.[1] ?? Math.max(...xs)) : data.length
    const xSpan = numX ? xMax - xMin : 1
    const xScale = (v: number, i: number): number => {
      if (!numX) return PAD.left + (i + 0.5) * (plotWidth / Math.max(1, data.length))
      return PAD.left + ((v - xMin) / (xSpan || 1)) * plotWidth
    }

    // y domain: explicit, else from data (plus any reference line).
    let yMin = 0
    let yMax = 0
    if (yDomain) {
      yMin = yDomain[0]
      yMax = yDomain[1]
    } else {
      const vals: number[] = []
      for (const row of data) {
        for (const s of series) {
          const v = row[s.key]
          if (typeof v === 'number' && Number.isFinite(v)) vals.push(v)
        }
      }
      if (referenceLine?.y !== undefined) vals.push(referenceLine.y)
      if (referenceLine?.from && referenceLine?.to) {
        vals.push(referenceLine.from[1], referenceLine.to[1])
      }
      if (vals.length) {
        yMin = Math.min(...vals)
        yMax = Math.max(...vals)
        const pad = (yMax - yMin) * 0.08 || 1
        yMin -= pad
        yMax += pad
      }
    }
    const ySpan = yMax - yMin || 1
    const yScale = (v: number): number => PAD.top + (1 - (v - yMin) / ySpan) * plotHeight

    const yTicks = niceTicks(yMin, yMax, 5)
    const xTicks = numX ? niceTicks(xMin, xMax, 5) : tickIndexes(data.length).map((i) => ({ i, v: data[i][xKey] }))

    // Line paths (break on undefined unless connectNulls).
    const lines = series.map((s) => {
      let path = ''
      let penDown = false
      for (let i = 0; i < data.length; i++) {
        const v = data[i][s.key]
        const defined = typeof v === 'number' && Number.isFinite(v)
        if (!defined) {
          penDown = false
          continue
        }
        const x = xScale(numX ? Number(data[i][xKey]) : 0, i)
        const y = yScale(v as number)
        if (!penDown) {
          path += `M ${x} ${y}`
          penDown = true
        } else if (s.connectNulls || path.length) {
          path += ` L ${x} ${y}`
        }
      }
      return { ...s, path }
    })

    return { plot: { width: plotWidth, height: plotHeight }, xScale, yScale, yTicks, xTicks, lines }
  }, [width, height, data, xKey, xType, xDomain, yDomain, series, referenceLine])

  /** Shared pointer logic: mouse (clientX) and touch (touches[0].clientX) both
   *  land here so tooltips behave identically on touch devices. */
  const handleMove = (clientX: number) => {
    const svg = svgRef.current
    if (!svg) return
    const rect = svg.getBoundingClientRect()
    const mx = clientX - rect.left
    if (mx < PAD.left || mx > width - PAD.right) {
      setHover(null)
      return
    }
    if (xType === 'number') {
      // nearest index by x value
      let best = 0
      let bestDist = Infinity
      for (let i = 0; i < data.length; i++) {
        const v = Number(data[i][xKey])
        const d = Math.abs(xScale(v, i) - mx)
        if (d < bestDist) {
          bestDist = d
          best = i
        }
      }
      setHover(best)
    } else {
      const slot = plot.width / Math.max(1, data.length)
      const idx = Math.floor((mx - PAD.left) / slot)
      setHover(Math.max(0, Math.min(data.length - 1, idx)))
    }
  }

  const onMouseMove = (e: MouseEvent<SVGSVGElement>) => handleMove(e.clientX)
  const onTouchMove = (e: TouchEvent<SVGSVGElement>) => {
    const touch = e.touches[0]
    if (touch) handleMove(touch.clientX)
  }

  const hoverRow = hover !== null ? data[hover] : undefined

  return (
    <div className="chart">
      <div ref={wrapRef} className="chart-canvas" style={{ height }}>
        {width > 0 && (
          <svg
            ref={svgRef}
            width={width}
            height={height}
            className="chart-svg"
            role="img"
            aria-label={ariaLabel}
            onMouseMove={onMouseMove}
            onMouseLeave={() => setHover(null)}
            onTouchStart={onTouchMove}
            onTouchMove={onTouchMove}
            onTouchEnd={() => setHover(null)}
          >
            {/* horizontal grid + y ticks */}
            {yTicks.map((t, i) => (
              <g key={`y-${i}`}>
                <line className="chart-grid-line" x1={PAD.left} y1={yScale(t)} x2={width - PAD.right} y2={yScale(t)} />
                <text className="chart-axis-label" x={PAD.left - 6} y={yScale(t) + 3} textAnchor="end">
                  {formatTick(t)}
                </text>
              </g>
            ))}
            {/* x ticks */}
            {xTicks.map((t, i) => {
              const tx = xType === 'number' ? xScale(t as number, 0) : xScale(0, (t as { i: number }).i)
              const label = xType === 'number' ? formatTick(t as number) : String((t as { v: string | number | undefined }).v)
              return (
                <g key={`x-${i}`}>
                  <line className="chart-grid-line" x1={tx} y1={PAD.top} x2={tx} y2={height - PAD.bottom} />
                  <text className="chart-axis-label" x={tx} y={height - PAD.bottom + 14} textAnchor="middle">
                    {label}
                  </text>
                </g>
              )
            })}
            {/* reference line */}
            {referenceLine && (
              <ReferenceLine
                spec={referenceLine}
                xScale={(v, i) => xScale(v, i)}
                yScale={yScale}
                pad={PAD}
                width={width}
              />
            )}
            {/* series lines */}
            {lines.map((s) => (
              <path
                key={s.key}
                d={s.path}
                fill="none"
                stroke={s.color}
                strokeWidth={s.strokeWidth ?? 1.5}
              />
            ))}
            {/* dots */}
            {series.flatMap((s) =>
              (s.dot ? data : []).flatMap((row, i) => {
                const v = row[s.key]
                if (typeof v !== 'number' || !Number.isFinite(v)) return []
                return (
                  <circle
                    key={`${s.key}-${i}`}
                    cx={xScale(xType === 'number' ? Number(row[xKey]) : 0, i)}
                    cy={yScale(v)}
                    r={3}
                    fill={s.color}
                  />
                )
              }),
            )}
            {/* hover crosshair */}
            {hover !== null && (
              <line
                className="chart-crosshair"
                x1={xScale(xType === 'number' ? Number(data[hover][xKey]) : 0, hover)}
                y1={PAD.top}
                x2={xScale(xType === 'number' ? Number(data[hover][xKey]) : 0, hover)}
                y2={height - PAD.bottom}
              />
            )}
          </svg>
        )}
        {hoverRow && hover !== null && (
          <div
            className="chart-tooltip"
            style={{
              left: Math.min(Math.max(xScale(xType === 'number' ? Number(data[hover][xKey]) : 0, hover) + 10, 8), Math.max(0, width - 140)),
              top: 8,
            }}
          >
            {series.map((s) => {
              const v = hoverRow[s.key]
              return (
                <div key={s.key} className="chart-tooltip-row">
                  <span className="chart-tooltip-swatch" style={{ background: s.color }} />
                  <span>{s.name}</span>
                  <span className="chart-tooltip-value">
                    {typeof v === 'number' ? valueFormat(v) : '–'}
                  </span>
                </div>
              )
            })}
          </div>
        )}
      </div>
      {series.length > 1 && (
        <div className="chart-legend">
          {series.map((s) => (
            <span key={s.key} className="chart-legend-item">
              <span className="chart-legend-swatch" style={{ background: s.color }} />
              {s.name}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function ReferenceLine({
  spec,
  xScale,
  yScale,
  pad,
  width,
}: {
  spec: ReferenceLineSpec
  xScale: (v: number, i: number) => number
  yScale: (v: number) => number
  pad: typeof PAD
  width: number
}) {
  if (spec.y !== undefined) {
    const y = yScale(spec.y)
    return (
      <g>
        <line className="chart-reference" x1={pad.left} y1={y} x2={width - pad.right} y2={y} />
        {spec.label && (
          <text className="chart-reference-label" x={width - pad.right} y={y - 4} textAnchor="end">
            {spec.label}
          </text>
        )}
      </g>
    )
  }
  if (spec.from && spec.to) {
    return (
      <line
        className="chart-reference"
        x1={xScale(spec.from[0], 0)}
        y1={yScale(spec.from[1])}
        x2={xScale(spec.to[0], 0)}
        y2={yScale(spec.to[1])}
      />
    )
  }
  return null
}

/** "Nice" ~n ticks across [min, max]. */
function niceTicks(min: number, max: number, count: number): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return []
  if (min === max) {
    min -= 1
    max += 1
  }
  const step = niceStep((max - min) / Math.max(1, count))
  const start = Math.ceil(min / step) * step
  const ticks: number[] = []
  for (let v = start; v <= max + 1e-9; v += step) {
    const rounded = Math.round(v * 1e6) / 1e6
    if (ticks.length === 0 || rounded !== ticks[ticks.length - 1]) ticks.push(rounded)
  }
  return ticks
}

function niceStep(raw: number): number {
  if (raw <= 0 || !Number.isFinite(raw)) return 1
  const mag = 10 ** Math.floor(Math.log10(raw))
  const norm = raw / mag
  let n: number
  if (norm < 1.5) n = 1
  else if (norm < 3.5) n = 2
  else if (norm < 7.5) n = 5
  else n = 10
  return n * mag
}

function formatTick(v: number): string {
  if (Math.abs(v) >= 1000) return v.toFixed(0)
  const rounded = Math.round(v * 1000) / 1000
  return String(rounded)
}

/** Downsample category x-axis labels to at most ~8 evenly spaced indexes. */
function tickIndexes(n: number): number[] {
  if (n <= 8) return Array.from({ length: n }, (_, i) => i)
  const step = Math.ceil(n / 8)
  const out: number[] = []
  for (let i = 0; i < n; i += step) out.push(i)
  return out
}
