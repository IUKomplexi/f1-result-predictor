import { useEffect, useState, type ReactNode } from 'react'
import {
  getConfig,
  putConfig,
  ApiError,
  type ConfigField,
  type ConfigResponse,
} from '../../api/client'
import { useApi } from '../../hooks/useApi'
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
  seasons: { min: number; max: number }
  paramsKeys: string[]
}

export function Settings({ onNavigate }: { onNavigate?: (tabId: string) => void }) {
  const { state, retry } = useApi('config', getConfig)
  if (state.phase === 'loading') return <Skeleton rows={6} />
  if (state.phase === 'error') return <ErrorState message={state.message} onRetry={retry} />
  return <SettingsForm data={state.data} onNavigate={onNavigate} />
}

function SettingsForm({
  data,
  onNavigate,
}: {
  data: ConfigResponse
  onNavigate?: (tabId: string) => void
}) {
  const [editor, setEditor] = useState<Editor>(() => fromResponse(data))
  const [status, setStatus] = useState<
    { kind: 'ok'; text: string } | { kind: 'error'; text: string } | null
  >(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    setEditor(fromResponse(data))
  }, [data])

  const setValue = (section: string, key: string, value: unknown) => {
    setEditor((ed) => ({
      ...ed,
      cfg: {
        ...ed.cfg,
        [section]: { ...ed.cfg[section], [key]: value },
      },
    }))
  }

  const setParam = (key: string, value: number) => {
    const params = { ...(editor.cfg.model?.params as Record<string, unknown> | undefined) }
    params[key] = value
    setValue('model', 'params', params)
  }

  const toggleFeature = (id: string, checked: boolean) => {
    const current = (editor.cfg.features?.enabled as string[] | null) ?? editor.defaults
    const next = checked ? [...current, id] : current.filter((f) => f !== id)
    setValue('features', 'enabled', next)
  }

  const save = async () => {
    setSaving(true)
    setStatus(null)
    try {
      await putConfig(editor.cfg)
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
                  onChange={(value) => setValue(section, field.key, value)}
                  editor={editor}
                  toggleFeature={toggleFeature}
                  setParam={setParam}
                  onNavigate={onNavigate}
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
  onNavigate,
}: {
  field: ConfigField
  value: unknown
  onChange: (value: unknown) => void
  editor: Editor
  toggleFeature: (id: string, checked: boolean) => void
  setParam: (key: string, value: number) => void
  onNavigate?: (tabId: string) => void
}) {
  const help = field.help ? <p className="field-help">{field.help}</p> : null

  if (field.key === 'enabled' && field.type === 'features') {
    const selectedIds = (editor.cfg.features?.enabled as string[] | null) ?? editor.defaults
    const groups: { key: string; label: string }[] = [
      { key: 'core', label: 'Core — on by default' },
      { key: 'selectable', label: 'Selectable — off by default' },
      { key: 'cut', label: 'Cut — removal improved the backtest' },
    ]
    return (
      <div className="field span-all">
        <span className="field-label">Features</span>
        <button type="button" className="link-button" onClick={reset(editor, onChange)}>
          Reset to registry defaults
        </button>
        {onNavigate ? (
          <button
            type="button"
            className="link-button"
            title="Open the Feature Lab tab to measure whether each feature actually helps walk-forward MAE/Spearman."
            onClick={() => onNavigate('features')}
          >
            Evaluate in Feature Lab
          </button>
        ) : null}
        {groups.map((group) => (
          <div key={group.key} className="feature-group">
            <h3 className="feature-group-title">{group.label}</h3>
            <div className="feature-grid">
              {editor.registry
                .filter((id) => editor.categories[id] === group.key)
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
