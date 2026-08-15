import { useEffect, useMemo, useReducer, useRef, useState, type ReactNode } from 'react'
import {
  ApiError,
  clearData,
  deleteModel,
  getConfig,
  getModels,
  putConfig,
  type ConfigField,
  type ConfigResponse,
  type ModelsResponse,
} from '../../api/client'
import type { NavState, TabProps } from '../../types'
import { useApi } from '../../hooks/useApi'
import { debounce, type Debounced } from '../../lib/debounce'
import { Badge } from '../../components/ui/Badge'
import { ConfirmDialog } from '../../components/ui/ConfirmDialog'
import { ErrorState, Skeleton } from '../../components/ui/DataState'
import { FeatureGroups } from '../../components/controls/FeatureGroups'
import { ModelParams } from '../../components/controls/ModelParams'
import { normPath } from '../../lib/models'
import './Settings.css'

type ConfigMap = Record<string, Record<string, unknown>>

/** Human-friendly labels for the raw [section]/key config identifiers. */
const SECTION_LABELS: Record<string, string> = {
  api: 'API',
  data: 'Data',
  model: 'Model',
  report: 'Reports',
  features: 'Features',
}

const KEY_LABELS: Record<string, string> = {
  base_url: 'Base URL',
  user_agent: 'User-Agent',
  sleep_seconds: 'Request spacing (seconds)',
  timeout: 'Timeout (seconds)',
  max_retries: 'Max retries',
  cache_dir: 'Raw cache directory',
  dataset: 'Dataset path',
  start_season: 'First season',
  end_season: 'Last season',
  checkpoint: 'Checkpoint path',
  calibrators: 'Calibrators path',
  seed: 'Random seed',
  params: 'Model hyperparameters',
  backtest: 'Backtest report path',
  prediction: 'Prediction report path',
  enabled: 'Feature selection',
}

/** A local editable copy of the effective config + registry metadata. */
interface Editor {
  cfg: ConfigMap
  schema: ConfigField[]
  registry: string[]
  defaults: string[]
  categories: Record<string, string>
  categoryMeta: { id: string; label: string }[]
  seasons: { min: number; max: number }
  paramsKeys: string[]
}

export function Settings(props: TabProps) {
  const { state, retry } = useApi('config', getConfig)
  if (state.phase === 'loading') return <Skeleton rows={6} />
  if (state.phase === 'error') return <ErrorState message={state.message} onRetry={retry} />
  return <SettingsForm data={state.data} {...props} />
}

type EditorAction =
  | { type: 'sync'; data: ConfigResponse }
  | { type: 'set-value'; section: string; key: string; value: unknown }
  | { type: 'toggle-feature'; id: string; checked: boolean }

function editorReducer(editor: Editor, action: EditorAction): Editor {
  switch (action.type) {
    case 'sync':
      return fromResponse(action.data)
    case 'set-value':
      return {
        ...editor,
        cfg: {
          ...editor.cfg,
          [action.section]: { ...editor.cfg[action.section], [action.key]: action.value },
        },
      }
    case 'toggle-feature': {
      const current = (editor.cfg.features?.enabled as string[] | null) ?? editor.defaults
      const next = action.checked
        ? [...current, action.id]
        : current.filter((f) => f !== action.id)
      return {
        ...editor,
        cfg: {
          ...editor.cfg,
          features: { ...editor.cfg.features, enabled: next },
        },
      }
    }
  }
}

function SettingsForm({
  data,
  onNavigate,
  setNavigateGuard,
}: { data: ConfigResponse } & TabProps) {
  const [editor, dispatch] = useReducer(editorReducer, data, fromResponse)
  const [status, setStatus] = useState<
    { kind: 'ok'; text: string } | { kind: 'error'; text: string } | null
  >(null)
  const [saving, setSaving] = useState(false)
  const [savedCfg, setSavedCfg] = useState<ConfigMap>(() => structuredClone(data.config) as ConfigMap)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const modelsState = useApi<ModelsResponse>('models', getModels)
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null)
  const [clearOpen, setClearOpen] = useState(false)
  const [manageStatus, setManageStatus] = useState<{ kind: 'ok' | 'error'; text: string } | null>(null)
  const pendingNav = useRef<{ id: string; state?: NavState } | null>(null)
  const editorRef = useRef(editor)
  editorRef.current = editor

  useEffect(() => {
    dispatch({ type: 'sync', data })
  }, [data])

  const dirty = JSON.stringify(editor.cfg) !== JSON.stringify(savedCfg)
  const dirtyRef = useRef(dirty)
  dirtyRef.current = dirty

  // Veto tab switches while edits are unsaved; the confirm dialog retries the
  // pending navigation through onNavigate after the user discards. Navigating
  // back to Settings itself is always allowed (nothing is lost).
  useEffect(() => {
    setNavigateGuard?.((id) => {
      if (id === 'settings' || !dirtyRef.current) return true
      pendingNav.current = { id }
      setConfirmOpen(true)
      return false
    })
    return () => setNavigateGuard?.(null)
  }, [setNavigateGuard])

  const setValue = (section: string, key: string, value: unknown) => {
    dispatch({ type: 'set-value', section, key, value })
  }

  const toggleFeature = (id: string, checked: boolean) => {
    dispatch({ type: 'toggle-feature', id, checked })
  }

  // Text/number inputs are debounced so rapid typing does not re-render the
  // whole form per keystroke; save() flushes pending edits first so no
  // keystroke is lost.
  const debouncedSetters = useMemo(() => new Map<string, Debounced<[unknown]>>(), [])
  const debouncedSetValue = (section: string, key: string) => {
    const id = `${section}.${key}`
    let setter = debouncedSetters.get(id)
    if (!setter) {
      setter = debounce((value: unknown) => {
        dispatch({ type: 'set-value', section, key, value })
      }, 300)
      debouncedSetters.set(id, setter)
    }
    return (value: unknown) => setter(value)
  }
  const flushAll = () => {
    for (const setter of debouncedSetters.values()) setter.flush()
  }

  const save = async () => {
    setSaving(true)
    setStatus(null)
    try {
      flushAll()
      // Let the flushed dispatches re-render so the saved config includes
      // everything the user typed.
      await new Promise((resolve) => setTimeout(resolve, 0))
      await putConfig(editorRef.current.cfg)
      setSavedCfg(structuredClone(editorRef.current.cfg) as ConfigMap)
      setStatus({ kind: 'ok', text: 'Saved. Retrain if you changed features.' })
    } catch (error) {
      const message = error instanceof ApiError ? error.message : String(error)
      setStatus({ kind: 'error', text: message })
    } finally {
      setSaving(false)
    }
  }

  const sections = [...new Set(editor.schema.map((f) => f.section))]
  const deployedPath = (editor.cfg.model?.checkpoint as string | undefined) ?? null
  const isDeployedModel = (name: string): boolean => {
    if (modelsState.state.phase !== 'ready' || deployedPath === null) return false
    const info = modelsState.state.data.models[name]
    return info !== undefined && normPath(info.checkpoint) === normPath(deployedPath)
  }
  const deleteTargetIsDeployed = deleteTarget !== null && isDeployedModel(deleteTarget)

  return (
    <>
      <section className="card">
        <h2 className="card-title">Configuration</h2>
        <p className="muted config-intro">
          These settings are saved to <code>config.toml</code> and used
          everywhere (web and CLI). Changing features means you must retrain
          the model. Hyperparameters are set per training on the Train tab.
        </p>
        {sections.map((section) => (
          <SectionGroup key={section} title={section}>
            {editor.schema
              .filter((f) => f.section === section)
              .map((field) => (
                <Field
                  key={field.key}
                  field={field}
                  value={editor.cfg[section]?.[field.key]}
                  onChange={
                    field.type === 'str' || field.type === 'int' || field.type === 'float'
                      ? debouncedSetValue(section, field.key)
                      : (value) => setValue(section, field.key, value)
                  }
                  editor={editor}
                  toggleFeature={toggleFeature}
                />
              ))}
          </SectionGroup>
        ))}
        <div className="save-bar">
          {dirty ? (
            <Badge variant="warn">Unsaved changes</Badge>
          ) : (
            <span className="muted">All changes saved</span>
          )}
          {status ? (
            <p className={status.kind === 'ok' ? 'save-status ok' : 'save-status error'} role="status">
              {status.text}
            </p>
          ) : null}
          <button
            type="button"
            className="button primary"
            onClick={save}
            disabled={saving || !dirty}
          >
            {saving ? 'Saving…' : 'Save configuration'}
          </button>
        </div>
      </section>
      <section className="card">
        <h2 className="card-title">Models</h2>
        <p className="muted config-intro">
          Every trained model. Deleting the deployed model (the config
          default) is allowed, but predictions need a retrain before they work
          again.
        </p>
        {modelsState.state.phase === 'loading' ? (
          <Skeleton rows={2} />
        ) : modelsState.state.phase === 'error' ? (
          <ErrorState message={modelsState.state.message} onRetry={modelsState.retry} />
        ) : (
          <ModelList
            models={modelsState.state.data}
            deployedPath={(editor.cfg.model?.checkpoint as string | undefined) ?? null}
            onDelete={(name) => setDeleteTarget(name)}
          />
        )}
        {manageStatus ? (
          <p
            className={manageStatus.kind === 'ok' ? 'save-status ok' : 'save-status error'}
            role="status"
          >
            {manageStatus.text}
          </p>
        ) : null}
      </section>
      <section className="card danger-zone">
        <h2 className="card-title">Clear data</h2>
        <p className="muted config-intro">
          Removes everything the app generated: the dataset, all trained
          models, predictions and reports. The downloaded raw data (
          <code>data/raw</code>) is kept, so you can retrain without
          re-fetching.
        </p>
        <button type="button" className="button" onClick={() => setClearOpen(true)}>
          Clear all generated data
        </button>
      </section>
      {confirmOpen ? (
        <ConfirmDialog
          title="Discard unsaved changes?"
          body="Your configuration edits have not been saved. Leaving the Settings tab discards them."
          confirmLabel="Discard changes"
          cancelLabel="Keep editing"
          onConfirm={() => {
            setConfirmOpen(false)
            const pending = pendingNav.current
            pendingNav.current = null
            // Release the guard first so the retry navigation is not vetoed
            // again (the editor state becomes irrelevant: the tab unmounts).
            dirtyRef.current = false
            if (pending) onNavigate?.(pending.id)
          }}
          onCancel={() => {
            setConfirmOpen(false)
            pendingNav.current = null
            // Browser back/forward may already have moved the hash; snap it
            // back without pushing another history entry.
            history.replaceState(null, '', '#/settings')
          }}
        />
      ) : null}
      {deleteTarget !== null ? (
        <ConfirmDialog
          title={
            deleteTargetIsDeployed
              ? `Delete the deployed model ${deleteTarget}?`
              : `Delete model ${deleteTarget}?`
          }
          body={
            deleteTargetIsDeployed
              ? 'This is the config-default model predictions currently use. Deleting it means "retrain needed" until you train a replacement.'
              : 'Removes the checkpoint, its calibrators and the model index entry. This cannot be undone.'
          }
          confirmLabel="Delete model"
          cancelLabel="Cancel"
          onConfirm={async () => {
            const name = deleteTarget
            setDeleteTarget(null)
            try {
              await deleteModel(name)
              modelsState.retry()
              setManageStatus({ kind: 'ok', text: `Deleted ${name}.` })
            } catch (error) {
              const message = error instanceof ApiError ? error.message : String(error)
              setManageStatus({ kind: 'error', text: message })
            }
          }}
          onCancel={() => setDeleteTarget(null)}
        />
      ) : null}
      {clearOpen ? (
        <ConfirmDialog
          title="Clear all generated data?"
          body="This removes the dataset, all trained models, predictions and reports. The downloaded raw data (data/raw) is kept. You will need to retrain before predicting again."
          confirmLabel="Clear everything"
          cancelLabel="Cancel"
          onConfirm={async () => {
            setClearOpen(false)
            try {
              const result = await clearData()
              modelsState.retry()
              setManageStatus({
                kind: 'ok',
                text: `Cleared ${Object.values(result.removed).reduce((a, b) => a + b, 0)} files — retrain from the Train tab.`,
              })
            } catch (error) {
              const message = error instanceof ApiError ? error.message : String(error)
              setManageStatus({ kind: 'error', text: message })
            }
          }}
          onCancel={() => setClearOpen(false)}
        />
      ) : null}
    </>
  )
}

function ModelList({
  models,
  deployedPath,
  onDelete,
}: {
  models: ModelsResponse
  deployedPath: string | null
  onDelete: (name: string) => void
}) {
  const entries = Object.entries(models.models).sort(([a], [b]) => a.localeCompare(b))
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col">Checkpoint</th>
            <th scope="col" className="num">Features</th>
            <th scope="col" className="num">Seasons</th>
            <th scope="col" className="num">Trained</th>
            <th scope="col" />
          </tr>
        </thead>
        <tbody>
          {entries.map(([name, info]) => {
            const isDeployed =
              deployedPath !== null && normPath(info.checkpoint) === normPath(deployedPath)
            return (
              <tr key={name}>
                <td>
                  <code className="mono">{name}</code>{' '}
                  {isDeployed ? <Badge variant="info">deployed</Badge> : null}
                </td>
                <td className="muted">{info.checkpoint}</td>
                <td className="num">{info.features?.length ?? '–'}</td>
                <td className="num">
                  {info.season_range ? info.season_range.join('–') : '–'}
                </td>
                <td className="num">
                  {info.trained_at
                    ? new Date(info.trained_at * 1000).toLocaleDateString()
                    : '–'}
                </td>
                <td>
                  <button type="button" className="button" onClick={() => onDelete(name)}>
                    Delete
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function fromResponse(data: ConfigResponse): Editor {
  return {
    cfg: structuredClone(data.config) as ConfigMap,
    schema: data.schema,
    registry: data.features.registry,
    defaults: data.features.defaults,
    categories: data.features.categories,
    categoryMeta: data.features.category_meta,
    seasons: data.seasons,
    paramsKeys: data.model_params_keys,
  }
}

function SectionGroup({ title, children }: { title: string; children: ReactNode }) {
  return (
    <fieldset className="config-section">
      <legend>
        {SECTION_LABELS[title] ?? title}{' '}
        <code className="section-key">[{title}]</code>
      </legend>
      <div className="field-grid">{children}</div>
    </fieldset>
  )
}

function fieldLabel(field: ConfigField): string {
  return KEY_LABELS[field.key] ?? field.key
}

function Field({
  field,
  value,
  onChange,
  editor,
  toggleFeature,
}: {
  field: ConfigField
  value: unknown
  onChange: (value: unknown) => void
  editor: Editor
  toggleFeature: (id: string, checked: boolean) => void
}) {
  const help = field.help ? <p className="field-help">{field.help}</p> : null

  if (field.key === 'enabled' && field.type === 'features') {
    const selectedIds = (editor.cfg.features?.enabled as string[] | null) ?? editor.defaults
    return (
      <div className="field span-all">
        <FeatureGroups
          registry={editor.registry}
          categories={editor.categories}
          categoryMeta={editor.categoryMeta}
          checked={(id) => selectedIds.includes(id)}
          onToggle={toggleFeature}
          resetLabel="Reset to registry defaults"
          onReset={reset(editor, onChange)}
          hint={help}
        />
      </div>
    )
  }

  if (field.key === 'params' && field.type === 'params') {
    const params = (value as Record<string, number> | undefined) ?? null
    return (
      <div className="field span-all">
        <span className="field-label">Model hyperparameters</span>
        <ModelParams
          keys={editor.paramsKeys}
          value={params}
          disabled
          hint="Locked: hyperparameters are set per training on the Train tab — the deployed model keeps the params it was trained with."
        />
        {help}
      </div>
    )
  }

  if (field.type === 'int' || field.type === 'float') {
    const kind: 'int' | 'float' = field.type
    return (
      <div className="field">
        <div className="field-header">
          <label className="field-label" htmlFor={`${field.section}-${field.key}`}>
            {fieldLabel(field)}
          </label>
          <code className="field-key">[{field.section}] {field.key}</code>
        </div>
        <input
          id={`${field.section}-${field.key}`}
          type="number"
          step={kind === 'float' ? 'any' : '1'}
          min={field.min}
          max={field.max}
          value={String(value ?? '')}
          onChange={(e) => onChange(num(kind, e.target.value))}
        />
        {help}
      </div>
    )
  }

  return (
    <div className="field">
      <div className="field-header">
        <label className="field-label" htmlFor={`${field.section}-${field.key}`}>
          {fieldLabel(field)}
        </label>
        <code className="field-key">[{field.section}] {field.key}</code>
      </div>
      <input
        id={`${field.section}-${field.key}`}
        type="text"
        value={String(value ?? '')}
        onChange={(e) => onChange(e.target.value)}
      />
      {help}
    </div>
  )
}

function num(kind: 'int' | 'float', text: string): number | string {
  if (text === '') return ''
  const n = kind === 'int' ? parseInt(text, 10) : parseFloat(text)
  return Number.isNaN(n) ? text : n
}

function reset(editor: Editor, onChange: (value: unknown) => void) {
  return () => onChange(editor.defaults)
}
