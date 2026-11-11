import { parquetMetadataAsync, parquetReadObjects, asyncBufferFromUrl } from 'hyparquet'
import { compressors } from 'hyparquet-compressors'
import './data.css'

// Browse the parquet files of a Hugging Face dataset (default: tiny-rl-refined-tasks-v5); ?repo=owner/name for another.

const REPO = new URLSearchParams(location.search).get('repo') || 'davidheineman/tiny-rl-refined-tasks-v5'
const BASE = `https://huggingface.co/datasets/${REPO}/resolve/main/`
const $ = (id) => document.getElementById(id)
const els = { split: $('split'), search: $('search'), list: $('list'), detail: $('detail'), tasks: $('tasks'), configs: $('configs') }
$('title').textContent = REPO
$('title').href = `https://huggingface.co/datasets/${REPO}`
document.title = `${REPO} viewer`

// tree: config -> split -> task -> [paths]
const tree = {}
const state = { config: '', split: '', task: '' }
let rows = []
let filtered = []
let selected = -1
let loadToken = 0
let loading = null // { loaded, total } while a task is loading
const LIST_LIMIT = 500
let listShown = LIST_LIMIT

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c])
}
const toJSON = (v) => JSON.stringify(v, (_, x) => (typeof x === 'bigint' ? Number(x) : x), 2)
const pick = (want, opts) => (opts.includes(want) ? want : opts[0])
const splitsOf = (config) => Object.keys(tree[config] ?? {})
const tasksOf = (config, split) => Object.keys(tree[config]?.[split] ?? {}).sort()
const pathsOf = () => tree[state.config]?.[state.split]?.[state.task] ?? []

// Configs come from the dataset card (README `configs:` block), so new configs show up automatically.
// Each data_files glob is matched against the repo's file list; the task is the file's parent directory.
// Falls back to scanning data/<config>/[<split>/]<task>/*.parquet when the card defines no configs.
const globToRe = (g) => new RegExp('^' + g.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*\*/g, '\0').replace(/\*/g, '[^/]*').replace(/\0/g, '.*') + '$')

async function loadTree() {
  const info = await (await fetch(`https://huggingface.co/api/datasets/${REPO}`)).json()
  const files = info.siblings.map((s) => s.rfilename).filter((f) => f.endsWith('.parquet')).sort()
  const configs = info.cardData?.configs ?? []
  for (const c of configs) {
    const dataFiles = typeof c.data_files === 'string' ? [{ split: 'train', path: c.data_files }] : c.data_files ?? []
    for (const df of dataFiles) {
      for (const glob of [df.path].flat()) {
        const re = globToRe(glob)
        for (const f of files.filter((f) => re.test(f))) {
          const task = f.split('/').at(-2)
          ;(((tree[c.config_name] ??= {})[df.split ?? 'train'] ??= {})[task] ??= []).push(f)
        }
      }
    }
  }
  if (!configs.length) {
    for (const f of files.filter((f) => f.startsWith('data/'))) {
      const parts = f.split('/').slice(1)
      const split = parts.length >= 4 ? parts[1] : 'train'
      ;(((tree[parts[0]] ??= {})[split] ??= {})[parts.at(-2)] ??= []).push(f)
    }
  }
  readHash()
}

// Resolve config/split/task (falling back to valid values), refresh the controls, and load the task.
function setSelection({ config = state.config, split = state.split, task = state.task }, selectRow = 0) {
  state.config = pick(config, Object.keys(tree))
  state.split = pick(split, splitsOf(state.config))
  state.task = pick(task, tasksOf(state.config, state.split))
  els.split.innerHTML = splitsOf(state.config).map((s) => `<option>${esc(s)}</option>`).join('')
  els.split.value = state.split
  renderConfigs()
  renderTasks()
  loadTask(selectRow)
}

function readHash() {
  const h = Object.fromEntries(new URLSearchParams(location.hash.slice(1)))
  setSelection(h, h.row ? +h.row : 0)
}

function writeHash() {
  const h = new URLSearchParams(state)
  if (selected >= 0) h.set('row', selected)
  history.replaceState(null, '', '#' + h)
}

els.split.onchange = () => setSelection({ split: els.split.value })

// ---------- task + config panels ----------
const rowCounts = {} // path -> Promise<num_rows>
function countRows(path) {
  return (rowCounts[path] ??= asyncBufferFromUrl({ url: BASE + path }).then(parquetMetadataAsync).then((m) => Number(m.num_rows)))
}
function panelHTML(names, current) {
  return names.map((n) => `<div class="t${n === current ? ' sel' : ''}" data-t="${esc(n)}"><span>${esc(n)}</span><span class="c"></span></div>`).join('')
}
function renderConfigs() {
  els.configs.innerHTML = panelHTML(Object.keys(tree), state.config)
}
function renderTasks() {
  const byTask = tree[state.config]?.[state.split] ?? {}
  const names = tasksOf(state.config, state.split)
  els.tasks.innerHTML = panelHTML(names, state.task)
  for (const t of names) {
    Promise.all(byTask[t].map(countRows))
      .then((ns) => {
        const el = els.tasks.querySelector(`.t[data-t="${CSS.escape(t)}"] .c`)
        if (el) el.textContent = ns.reduce((a, b) => a + b, 0).toLocaleString()
      })
      .catch(() => {})
  }
}
els.tasks.onclick = (e) => {
  const t = e.target.closest('.t')?.dataset.t
  if (t && t !== state.task) setSelection({ task: t })
}
els.configs.onclick = (e) => {
  const c = e.target.closest('.t')?.dataset.t
  if (c && c !== state.config) setSelection({ config: c })
}

// Load every part of the current task, row group by row group, so the list fills progressively.
async function loadTask(selectRow = 0) {
  const token = ++loadToken
  const paths = pathsOf()
  rows = []
  filtered = []
  selected = -1
  els.list.innerHTML = ''
  els.detail.innerHTML = '<p class="muted">Loading…</p>'
  try {
    const files = await Promise.all(
      paths.map(async (path) => {
        const file = await asyncBufferFromUrl({ url: BASE + path })
        return { file, metadata: await parquetMetadataAsync(file) }
      })
    )
    if (token !== loadToken) return
    loading = { total: files.reduce((a, f) => a + Number(f.metadata.num_rows), 0) }
    let offset = 0
    for (const { file, metadata } of files) {
      let start = 0
      for (const rg of metadata.row_groups) {
        const end = start + Number(rg.num_rows)
        const chunk = await parquetReadObjects({ file, metadata, rowStart: start, rowEnd: end, compressors })
        if (token !== loadToken) return
        chunk.forEach((r, i) => (r.__i = offset + start + i))
        rows.push(...chunk)
        start = end
        applyFilter(false)
        if (selected < 0 && rows.length > selectRow) select(selectRow)
        window.__ready = true // for scripted tests
      }
      offset += start
    }
    loading = null
    applyFilter(false)
  } catch (e) {
    if (token !== loadToken) return
    loading = null
    els.detail.innerHTML = `<p class="muted">Error loading ${esc(state.task)}: ${esc(e.message)}</p>`
    console.error(e)
  }
}

// ---------- search ----------
let searchTimer
els.search.oninput = () => {
  clearTimeout(searchTimer)
  searchTimer = setTimeout(() => applyFilter(true), 150)
}
function searchRe() {
  const q = els.search.value.trim()
  if (!q) return null
  try {
    return new RegExp(q, 'gi')
  } catch {
    return new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi')
  }
}
function applyFilter(resetScroll) {
  const re = searchRe()
  filtered = re
    ? rows.filter((r) => Object.values(r).some((v) => typeof v === 'string' && ((re.lastIndex = 0), re.test(v))))
    : rows
  if (resetScroll) {
    listShown = LIST_LIMIT
    els.list.scrollTop = 0
  }
  renderList()
  if (resetScroll && selected >= 0) renderDetail()
}

// ---------- list ----------
function snippet(r) {
  const s = r.question || r.text || r.prompt || ''
  return s.replace(/\s+/g, ' ').slice(0, 160)
}
function renderList() {
  const shown = filtered.slice(0, listShown)
  els.list.innerHTML =
    shown
      .map((r) => {
        const id = r.source_id ?? r.document_id ?? ''
        const meta = [`#${r.__i}`, id, r.hint_level != null ? `hint ${r.hint_level}` : '', r.tokens != null ? `${r.tokens} tok` : '', r.example_count != null ? `${r.example_count} ex` : '']
          .filter(Boolean)
          .join(' · ')
        return `<div class="row${r.__i === selected ? ' sel' : ''}" data-i="${r.__i}"><div class="meta">${esc(meta)}</div><div class="snip">${esc(snippet(r))}</div></div>`
      })
      .join('') +
    `<div class="pager">${loading ? `loading… ${rows.length.toLocaleString()} / ${loading.total.toLocaleString()} rows · ` : ''}${filtered.length.toLocaleString()} matching${filtered.length > listShown ? ` · showing ${listShown} (scroll for more)` : ''}</div>`
}
els.list.onclick = (e) => {
  const el = e.target.closest('.row')
  if (el) select(+el.dataset.i)
}
els.list.onscroll = () => {
  if (els.list.scrollTop + els.list.clientHeight > els.list.scrollHeight - 200 && listShown < filtered.length) {
    listShown += LIST_LIMIT
    renderList()
  }
}
document.onkeydown = (e) => {
  if (e.target === els.search || !['ArrowDown', 'ArrowUp', 'j', 'k'].includes(e.key)) return
  e.preventDefault()
  const idx = filtered.findIndex((r) => r.__i === selected)
  const next = filtered[Math.max(0, Math.min(filtered.length - 1, idx + (e.key === 'ArrowDown' || e.key === 'j' ? 1 : -1)))]
  if (next) {
    select(next.__i)
    els.list.querySelector('.row.sel')?.scrollIntoView({ block: 'nearest' })
  }
}

function select(i) {
  selected = i
  els.list.querySelectorAll('.row.sel').forEach((el) => el.classList.remove('sel'))
  els.list.querySelector(`.row[data-i="${i}"]`)?.classList.add('sel')
  renderDetail()
  writeHash()
}

// ---------- detail ----------
const SCALARS = ['task', 'source_id', 'document_id', 'source_split', 'hint_level', 'prompt_tokens', 'completion_tokens', 'tokens', 'budget_tokens', 'example_count', 'version', 'group']

function hl(text) {
  const re = searchRe()
  if (!re) return esc(text)
  let out = ''
  let last = 0
  for (const m of text.matchAll(re)) {
    if (!m[0]) break
    out += esc(text.slice(last, m.index)) + `<mark class="hl">${esc(m[0])}</mark>`
    last = m.index + m[0].length
  }
  return out + esc(text.slice(last))
}

function prettyMaybeJSON(v) {
  if (typeof v !== 'string') return toJSON(v)
  const t = v.trim()
  if ((t.startsWith('{') || t.startsWith('[')) && t.length > 1) {
    try {
      return JSON.stringify(JSON.parse(t), null, 2)
    } catch {}
  }
  return v
}

function field(name, body, note = '') {
  return `<div class="field"><h4>${esc(name)} <span class="n">${esc(note)}</span></h4>${body}</div>`
}
function textField(name, v) {
  if (v == null || v === '' || v === '[]' || v === '{}') return field(name, '<pre class="empty">(empty)</pre>')
  const s = prettyMaybeJSON(v)
  return field(name, `<pre>${hl(s)}</pre>`, `${s.length.toLocaleString()} chars`)
}

function renderDetail() {
  const r = rows.find((x) => x.__i === selected)
  if (!r) return
  let html = '<div class="kv">' + SCALARS.filter((k) => k in r).map((k) => `<span>${k}</span><span>${esc(r[k])}</span>`).join('') + '</div>'

  if ('prompt' in r && 'completion' in r) {
    html += field(
      'model view: prompt → completion',
      `<pre><span class="p-prompt">${hl(r.prompt)}</span><span class="p-completion">${hl(r.completion)}</span></pre>`,
      'grey = prompt, yellow = completion'
    )
  }

  const order = ['question', 'text', 'answer', 'full_reference', 'tests', 'setup', 'keywords', 'rejected_reply', 'baseline', 'completion', 'completion_ids', 'source_ids', 'provenance', 'prefix', 'prompt', 'demonstrations']
  const rest = Object.keys(r).filter((k) => k !== '__i' && !SCALARS.includes(k) && !order.includes(k))
  for (const k of [...order.filter((k) => k in r), ...rest]) {
    if (k === 'demonstrations') {
      let demos
      try {
        demos = JSON.parse(r[k])
      } catch {}
      if (Array.isArray(demos) && demos.length) {
        html += field(
          'demonstrations',
          demos.map((d, j) => `<div class="kv" style="margin:6px 0 2px"><span>#${j}</span><span>${esc(d.source_id ?? '')} · ${esc(d.split ?? '')}</span></div><pre>${hl(d.text ?? toJSON(d))}</pre>`).join(''),
          `${demos.length} demos`
        )
        continue
      }
    }
    html += textField(k, r[k])
  }
  els.detail.innerHTML = html
  els.detail.scrollTop = 0
}

window.addEventListener("hashchange", readHash)
loadTree().catch((e) => (els.detail.innerHTML = `<p class="muted">Error listing files: ${esc(e.message)}</p>`))
