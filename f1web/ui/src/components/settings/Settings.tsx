import { useEffect, useMemo, useReducer, useRef, useState, type ReactNode } from 'react'
import {
  getConfig,
  putConfig,
  ApiError,
  type ConfigField,
  type ConfigResponse,
} from '../../api/client'
import type { NavState, TabProps } from '../../App'
import { useApi } from '../../hooks/useApi'
import { debounce, type Debounced } from '../../lib/debounce'
import { Badge } from '../ui/Badge'
import { ConfirmDialog } from '../ui/ConfirmDialog'
import { ErrorState, Skeleton } from '../ui/DataState'
import { FeatureGroups } from '../ui/FeatureGroups'
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
  | { type: 'set-param'; key: string; value: number }
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
    case 'set-param': {
      const params = { ...(editor.cfg.model?.params as Record<string, unknown> | undefined) }
      params[action.key] = action.value
      return {
        ...editor,
        cfg: {
          ...editor.cfg,
          model: { ...editor.cfg.model, params },
        },
      }
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

  const setParam = (key: string, value: number) => {
    dispatch({ type: 'set-param', key, value })
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
      setStatus({ kind: 'ok', text: 'Saved. Retrain if you changed features or model params.' })
    } catch (error) {
      const message = error instanceof ApiError ? error.message : String(error)
      setStatus({ kind: 'error', text: message })
    } finally {
      setSaving(false)
    }
  }

  const sections = [...new Set(editor.schema.map((f) => f.section))]

  return (
    <>
      <section className="card">
        <h2 className="card-title">Configuration</h2>
        <p className="muted config-intro">
          These settings are saved to <code>config.toml</code> and used
          everywhere (web and CLI). Changing features or model params means
          you must retrain the model.
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
                  setParam={setParam}
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
    </>
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
  setParam,
}: {
  field: ConfigField
  value: unknown
  onChange: (value: unknown) => void
  editor: Editor
  toggleFeature: (id: string, checked: boolean) => void
  setParam: (key: string, value: number) => void
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
    const params = (value as Record<string, unknown> | undefined) ?? {}
    return (
      <div className="field span-all">
        <span className="field-label">Model hyperparameters</span>
        <div className="param-grid">
          {editor.paramsKeys.map((key) => (
            <NumberField
              key={key}
              name={key}
              value={Number(params[key])}
              onChange={(v) => setParam(key, v)}
            />
          ))}
        </div>
        {help}
      </div>
    )
  }

  if (field.type === 'int' || field.type === 'float') {
    const kind: 'int' | 'float' = field.type
    return (
      <div className="field">
        <label className="field-label" htmlFor={`${field.section}-${field.key}`}>
          {fieldLabel(field)}
        </label>
        <code className="field-key">[{field.section}] {field.key}</code>
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
      <label className="field-label" htmlFor={`${field.section}-${field.key}`}>
        {fieldLabel(field)}
      </label>
      <code className="field-key">[{field.section}] {field.key}</code>
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

function NumberField({
  name,
  value,
  onChange,
}: {
  name: string
  value: number
  onChange: (value: number) => void
}) {
  return (
    <div className="field">
      <label className="field-label" htmlFor={`param-${name}`}>
        {name}
      </label>
      <input
        id={`param-${name}`}
        type="number"
        step="any"
        value={Number.isFinite(value) ? String(value) : ''}
        onChange={(e) => onChange(e.target.value === '' ? 0 : Number(e.target.value))}
      />
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
