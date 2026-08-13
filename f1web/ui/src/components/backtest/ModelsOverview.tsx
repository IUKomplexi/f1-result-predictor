import { fmtDate } from '../../lib/format'
import type { ModelInfo, ModelsResponse } from '../../api/client'
import { Badge } from '../ui/Badge'
import { Skeleton } from '../ui/DataState'
import { deployedName } from './lib'

/** Saved checkpoints at a glance (name, training window, rows, features). */
export function ModelsOverview({ models }: { models: ModelsResponse | null }) {
  const entries = models
    ? Object.entries(models.models).sort(([a], [b]) => a.localeCompare(b))
    : []
  const deployed = deployedName(models)
  return (
    <section className="card">
      <h2 className="card-title">Saved models</h2>
      {models === null ? (
        <Skeleton rows={2} />
      ) : entries.length === 0 ? (
        <p className="muted">No saved models yet — name one on the Train tab.</p>
      ) : (
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Seasons</th>
                <th scope="col" className="num">Rows</th>
                <th scope="col" className="num">Features</th>
                <th scope="col" className="num">Params</th>
                <th scope="col">Trained</th>
              </tr>
            </thead>
            <tbody>
              {entries.map(([name, info]) => (
                <ModelRow key={name} name={name} info={info} deployed={deployed === name} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

function ModelRow({ name, info, deployed }: { name: string; info: ModelInfo; deployed: boolean }) {
  const trainedAt =
    typeof info.trained_at === 'number' ? fmtDate(new Date(info.trained_at * 1000).toISOString()) : '–'
  return (
    <tr className={deployed ? 'row-model' : undefined}>
      <td>
        {name}
        {deployed ? <Badge variant="info">deployed</Badge> : null}
      </td>
      <td>{info.season_range ? `${info.season_range[0]}–${info.season_range[1]}` : '–'}</td>
      <td className="num">{info.rows ?? '–'}</td>
      <td className="num">{info.features?.length ?? '–'}</td>
      <td className="num">{info.params ? Object.keys(info.params).length : '–'}</td>
      <td className="muted">{trainedAt}</td>
    </tr>
  )
}
