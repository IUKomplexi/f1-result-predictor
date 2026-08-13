import type { Prediction } from '../api/client'

/** One race's analyzed prediction-vs-actuals summary (Race History rows). */
export interface RaceResult {
  round: number
  raceName: string
  date: string
  predictedWinner: string
  actualWinner: string | null
  winnerHit: boolean | null
  top3Overlap: number | null
  top3Of: number
  hasActuals: boolean
}

/**
 * Reduce a race prediction to its Race History summary row: predicted vs
 * actual winner (hit?) and top-3 overlap. Pure — no React, no I/O — so it is
 * unit-testable and reusable outside the component tree.
 */
export function analyzeRace(prediction: Prediction): RaceResult {
  // Defensive: a malformed/partial prediction must degrade to an empty row,
  // not throw and take down the tab (see ErrorBoundary).
  const drivers = Array.isArray(prediction.drivers) ? prediction.drivers : []
  const race = prediction.race ?? { race_name: null, circuit_id: null, date: null }
  const sorted = [...drivers].sort((a, b) => a.pred_rank - b.pred_rank)
  const predictedTop3 = sorted.slice(0, 3).map((row) => row.driver_id)
  const predictedWinner = predictedTop3[0] ?? ''

  const raced = drivers.filter(
    (row) => row.actual_position !== null && row.actual_position !== undefined,
  )
  const winnerRow = raced.find((row) => row.actual_position === 1)
  if (!winnerRow) {
    return {
      round: prediction.round,
      raceName: race.race_name ?? `Round ${prediction.round}`,
      date: race.date ?? '',
      predictedWinner,
      actualWinner: null,
      winnerHit: null,
      top3Overlap: null,
      top3Of: 3,
      hasActuals: false,
    }
  }
  const actualByPosition = new Map(
    raced.map((row) => [row.actual_position as number, row.driver_id]),
  )
  const actualTop3 = [1, 2, 3]
    .map((pos) => actualByPosition.get(pos))
    .filter((id): id is string => id !== undefined)
  const overlap = predictedTop3.filter((id) => actualTop3.includes(id)).length
  return {
    round: prediction.round,
    raceName: race.race_name ?? `Round ${prediction.round}`,
    date: race.date ?? '',
    predictedWinner,
    actualWinner: actualByPosition.get(1) ?? null,
    winnerHit: actualByPosition.get(1) === predictedWinner,
    top3Overlap: overlap,
    top3Of: Math.min(3, actualTop3.length),
    hasActuals: true,
  }
}
