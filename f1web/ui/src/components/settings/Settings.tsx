import { useEffect, useMemo, useReducer, useRef, useState, type ReactNode } from 'react'
import {
  getConfig,
  putConfig,
  ApiError,
  type ConfigField,
  type ConfigResponse,
} from '../../api/client'
import { useApi } from '../../hooks/useApi'
import { debounce, type Debounced } from '../../lib/debounce'
import { Badge } from '../ui/Badge'
import { ErrorState, Skeleton } from '../ui/DataState'
import './Settings.css'

type ConfigMap = Record<string, Record<string, unknown>>

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

export function Settings() {
  const { state, retry } = useApi('config', getConfig)
  if (state.phase === 'loading') return <Skeleton rows={6} />
  if (state.phase === 'error') return <ErrorState message={state.message} onRetry={retry} />
  return <SettingsForm data={state.data} />
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

function SettingsForm({ data }: { data: ConfigResponse }) {
  const [editor, dispatch] = useReducer(editorReducer, data, fromResponse)
  const [status, setStatus] = useState<
    { kind: 'ok'; text: string } | { kind: 'error'; text: string } | null
  >(null)
  const [saving, setSaving] = useState(false)
  const editorRef = useRef(editor)
  editorRef.current = editor

  useEffect(() => {
    dispatch({ type: 'sync', data })
  }, [data])

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
      setStatus({ kind: 'ok', text: 'Saved to config.toml. Retrain the model if features/params changed.' })
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
          These values are written back to <code>config.toml</code>, the single
          source of truth shared with the CLI. Editing features or{' '}
          <code>[model.params]</code> requires a retrain.
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
        {status ? (
          <p className={status.kind === 'ok' ? 'save-status ok' : 'save-status error'} role="status">
            {status.text}
          </p>
        ) : null}
        <div className="save-row">
          <button type="button" className="button" onClick={save} disabled={saving}>
            {saving ? 'Saving…' : 'Save configuration'}
          </button>
        </div>
      </section>
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
      <legend>{title}</legend>
      <div className="field-grid">{children}</div>
    </fieldset>
  )
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
    // Groups come from the backend (features/registry.py CATEGORY_ORDER +
    // CATEGORY_LABELS via /api/config); unknown categories still render,
    // appended after the known ones so drift never hides a feature.
    const known = new Set(editor.categoryMeta.map((m) => m.id))
    const groups = [
      ...editor.categoryMeta,
      ...[...new Set(Object.values(editor.categories))]
        .filter((category) => !known.has(category))
        .map((category) => ({ id: category, label: category })),
    ]
    return (
      <div className="field span-all">
        <span className="field-label">Features</span>
        <button type="button" className="link-button" onClick={reset(editor, onChange)}>
          Reset to registry defaults
        </button>
        {groups.map((group) => (
          <div key={group.id} className="feature-group">
            <h3 className="feature-group-title">{group.label}</h3>
            <div className="feature-grid">
              {editor.registry
                .filter((id) => editor.categories[id] === group.id)
                .map((id) => {
                  const selected = selectedIds!.includes(id)
                  return (
                    <label key={id} className="feature-check">
                      <input
                        type="checkbox"
                        checked={selected}
                        onChange={(e) => toggleFeature(id, e.target.checked)}
                      />
                      <span>{id}</span>
                    </label>
                  )
                })}
            </div>
          </div>
        ))}
        {help}
      </div>
    )
  }

  if (field.section === 'weather') {
    // Weather is evaluated but not adopted (reports/weather.md): keep the cache
    // dir editable but present a visible, disabled "plug into data" placeholder.
    return (
      <div className="field span-all">
        <div className="weather-note">
          <Badge variant="warn">Not adopted</Badge>
          <p className="field-help">
            Weather features were evaluated but not adopted (see reports/weather.md);
            the shipped model does not use them. The data plumbing is not enabled.
          </p>
        </div>
        <label className="field-label" htmlFor={`${field.section}-${field.key}`}>
          {field.key}
        </label>
        <input
          id={`${field.section}-${field.key}`}
          type="text"
          value={String(value ?? '')}
          onChange={(e) => onChange(e.target.value)}
        />
        <button type="button" className="button" disabled title="Weather is not adopted; the data plumbing is not enabled.">
          Plug into data
        </button>
        {help}
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
          {field.key}
        </label>
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
        {field.key}
      </label>
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
