import { useEffect, useMemo, useState } from 'react'
import './App.css'

type Intent = 'flood' | 'no_flood' | 'visual_similarity' | 'spectral_water_signal' | 'spectral_water_increase' | 'spectral_water_decrease' | 'spectral_scene_similarity' | 'spectral_change_similarity' | 'unsupported'
type EOQuery = {
  intent: Intent; start_date?: string | null; end_date?: string | null
  place_name?: string | null; place_level?: 'country' | 'admin1' | null; place_country?: string | null
  requested_modality: 'rgb' | 's2_12band' | 'unspecified'; example_tile_id?: string | null
  unsupported_conditions: string[]; interpretation: string; original_query: string; resolved_place_id?: string | null
}
type Run = {
  run_id: string; architecture: string; modality: string; seed: number; gallery_count: number; test_query_count: number
  display_name: string; validation_macro_precision_at_5_mean: number; validation_macro_precision_at_5_std: number
  validation_seed_count: number; representative_seed_score: number; searchable_gallery_count: number; prototype_support_count: number
}
type Place = { place_id: string; name: string; level: 'country' | 'admin1'; country?: string | null }
type Tile = { tile_id: string; date?: string | null; sequence_id?: string | null }
type Evidence = { rank: number; tile_id: string; score: number; preview_url: string; timestamp?: string | null; latitude?: number | null; longitude?: number | null; sequence_id?: string | null; sensor?: string | null; bands: string[]; labels: string[] }
type SearchResponse = { query: EOQuery; run_id: string; retrieval_method: string; query_vector_method: string; similarity_measure: string; candidate_count: number; results: Evidence[]; notice?: string | null }
type IndexStats = { valid_pixels: number; mean: number; median: number; positive_fraction: number }
type SpectralEvidence = {
  rank: number; score: number; score_description: string; tile_id?: string | null
  before_tile_id?: string | null; after_tile_id?: string | null; before_date?: string | null; after_date?: string | null
  gap_days?: number | null; sequence_id?: string | null; latitude?: number | null; longitude?: number | null
  label?: string | null; preview_url?: string | null; before_preview_url?: string | null; after_preview_url?: string | null
  index_url?: string | null; before_index_url?: string | null; after_index_url?: string | null; change_map_url?: string | null
  index_summary: Record<string, IndexStats>; change_summary?: Record<string, IndexStats> | null
}
type SpectralSearchResponse = { query: EOQuery; retrieval_method: string; similarity_measure: string; candidate_count: number; results: SpectralEvidence[]; notice?: string | null }
type Health = { status: string; run_count: number; gazetteer_ready: boolean; dataset: { available_scenes: number; date_start: string; date_end: string } }

const api = async <T,>(path: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(`/api${path}`, { ...init, headers: { 'Content-Type': 'application/json', ...init?.headers } })
  if (!response.ok) { const payload = await response.json().catch(() => ({})); throw new Error(payload.detail ?? `Request failed (${response.status})`) }
  return response.json() as Promise<T>
}

function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const [queryText, setQueryText] = useState('Find scenes with a strong water-like spectral signal in Manicaland, Zimbabwe during 2019')
  const [runId, setRunId] = useState('')
  const [parsed, setParsed] = useState<EOQuery | null>(null)
  const [places, setPlaces] = useState<Place[]>([])
  const [placeId, setPlaceId] = useState('')
  const [beforeSearch, setBeforeSearch] = useState('')
  const [afterSearch, setAfterSearch] = useState('')
  const [beforeTiles, setBeforeTiles] = useState<Tile[]>([])
  const [afterTiles, setAfterTiles] = useState<Tile[]>([])
  const [beforeTile, setBeforeTile] = useState('')
  const [afterTile, setAfterTile] = useState('')
  const [count, setCount] = useState(10)
  const [results, setResults] = useState<SearchResponse | null>(null)
  const [spectralResults, setSpectralResults] = useState<SpectralSearchResponse | null>(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([api<Health>('/health'), api<Run[]>('/runs')]).then(([status, available]) => {
      setHealth(status); setRuns(available); if (available.length) setRunId(available[0].run_id)
    }).catch((reason: Error) => setError(reason.message))
  }, [])

  const selectedRun = useMemo(() => runs.find((run) => run.run_id === runId), [runs, runId])
  const requestedModalityMatches = !parsed || parsed.requested_modality === 'unspecified' || selectedRun?.modality === parsed.requested_modality
  const selectedPlace = places.find((place) => place.place_id === placeId)
  const isSpectral = Boolean(parsed?.intent.startsWith('spectral_'))

  async function parseQuery(event: React.FormEvent) {
    event.preventDefault(); setBusy('parse'); setError(''); setResults(null); setSpectralResults(null); setPlaces([]); setPlaceId(''); setBeforeTile(''); setAfterTile(''); setBeforeSearch(''); setAfterSearch(''); setBeforeTiles([]); setAfterTiles([])
    try {
      const response = await api<{ query: EOQuery }>('/parse', { method: 'POST', body: JSON.stringify({ query: queryText }) })
      setParsed(response.query)
      if (response.query.place_name && response.query.place_level) {
        const countryQuery = response.query.place_country ? `&country=${encodeURIComponent(response.query.place_country)}` : ''
        const found = await api<Place[]>(`/places?q=${encodeURIComponent(response.query.place_name)}&level=${response.query.place_level}${countryQuery}`)
        setPlaces(found); if (found.length === 1) setPlaceId(found[0].place_id)
      }
    } catch (reason) { setError((reason as Error).message) } finally { setBusy('') }
  }

  async function findTiles(value: string, side: 'before' | 'after') {
    if (side === 'before') setBeforeSearch(value); else setAfterSearch(value)
    if (value.trim().length < 3) { side === 'before' ? setBeforeTiles([]) : setAfterTiles([]); return }
    try {
      const found = await api<Tile[]>(`/tiles?q=${encodeURIComponent(value)}&limit=60`)
      side === 'before' ? setBeforeTiles(found) : setAfterTiles(found)
    } catch (reason) { setError((reason as Error).message) }
  }

  async function retrieve() {
    if (!parsed) return
    setBusy('retrieve'); setError(''); setResults(null); setSpectralResults(null)
    const requestQuery: EOQuery = { ...parsed, resolved_place_id: selectedPlace?.place_id ?? null }
    try {
      if (isSpectral) {
        const response = await api<SpectralSearchResponse>('/spectral/retrieve', { method: 'POST', body: JSON.stringify({ query: requestQuery, k: count, before_example_tile_id: beforeTile || null, after_example_tile_id: afterTile || null }) })
        setSpectralResults(response)
      } else {
        if (!runId) return
        requestQuery.example_tile_id = beforeTile || null
        setResults(await api<SearchResponse>('/retrieve', { method: 'POST', body: JSON.stringify({ query: requestQuery, run_id: runId, k: count }) }))
      }
    } catch (reason) { setError((reason as Error).message) } finally { setBusy('') }
  }

  const requiresBefore = parsed?.intent === 'spectral_scene_similarity' || parsed?.intent === 'spectral_change_similarity' || parsed?.intent === 'visual_similarity'
  const canRetrieve = Boolean(parsed && (isSpectral || (runId && requestedModalityMatches)) && parsed.intent !== 'unsupported' &&
    parsed.unsupported_conditions.length === 0 && (!requiresBefore || beforeTile) &&
    (parsed.intent !== 'spectral_change_similarity' || afterTile) && (!parsed.place_name || placeId))

  return <main className="app-shell">
    <header className="topbar"><a className="brand" href="#top"><span className="brand-mark">G</span><span>GeoRAG <small>EARTH OBSERVATION RETRIEVAL</small></span></a><div className="status"><span className={health?.status === 'ready' ? 'dot' : 'dot muted'} />{health ? `${health.dataset.available_scenes.toLocaleString()} scenes ready` : 'Connecting to local data'}</div></header>
    <section className="hero" id="top"><div className="eyebrow">M11 · SPECTRAL SIGNAL → EVIDENCE</div><h1>Ask the archive<br/><em>for evidence.</em></h1><p>Search Sentinel-2 scenes with multispectral water and vegetation signals. Compare dates from the same location and inspect the index maps behind every spectral ranking.</p><div className="coverage"><span>LOCAL SEN12-FLOOD ARCHIVE</span><b>{health?.dataset.date_start ?? '—'} <i>→</i> {health?.dataset.date_end ?? '—'}</b><span>12-BAND SOURCE · {health?.run_count ?? 0} RGB MODEL BASELINES</span></div></section>

    <section className="workspace"><div className="search-panel">
      <div className="panel-title"><span className="step">01</span><div><h2>Describe a retrieval</h2><p>Ask for spectral evidence or an RGB embedding comparison.</p></div></div>
      <form onSubmit={parseQuery}><label htmlFor="query">NATURAL-LANGUAGE REQUEST</label><textarea id="query" value={queryText} onChange={(event) => { setQueryText(event.target.value); setParsed(null); setResults(null); setSpectralResults(null) }} rows={3} maxLength={1000}/><div className="helper">Try “find scenes with a strong water-like signal”, “find locations where water-like signal increased”, or compare spectral patterns with a selected scene or date pair. Flood labels remain scene-level.</div><button className="primary" disabled={busy === 'parse' || !health?.gazetteer_ready}>{busy === 'parse' ? 'Parsing request…' : 'Parse request'} <span>↗</span></button>{!health?.gazetteer_ready && <div className="warning">Prepare the offline place database with <code>uv run python scripts/prepare_natural_earth.py</code>.</div>}</form>
      {parsed && <div className="parsed-card"><div className="parsed-heading"><span className="check">✓</span><div><strong>Parsed request</strong><small>Review before searching</small></div></div><p className="interpretation">{parsed.interpretation}</p><div className="chips"><span>{parsed.intent.replaceAll('_', ' ')}</span>{parsed.start_date && <span>from {parsed.start_date}</span>}{parsed.end_date && <span>through {parsed.end_date}</span>}{parsed.place_name && <span>{parsed.place_name}{parsed.place_country ? `, ${parsed.place_country}` : ''} · {parsed.place_level}</span>}{parsed.requested_modality !== 'unspecified' && <span>{parsed.requested_modality}</span>}</div>{parsed.unsupported_conditions.length > 0 && <div className="warning">Unsupported: {parsed.unsupported_conditions.join('; ')}</div>}{parsed.place_name && <div className="field"><label htmlFor="place">MATCH THE PLACE</label>{places.length ? <select id="place" value={placeId} onChange={(event) => setPlaceId(event.target.value)}><option value="">Choose the correct place…</option>{places.map((place) => <option key={place.place_id} value={place.place_id}>{place.name}{place.country ? ` — ${place.country}` : ''}</option>)}</select> : <p className="helper">No matching country or first-order region found in the offline gazetteer.</p>}</div>}</div>}
    </div>

    <aside className="controls-panel"><div className="panel-title"><span className="step">02</span><div><h2>{isSpectral ? 'Choose spectral examples' : 'Choose the search space'}</h2><p>{isSpectral ? 'Ranking uses Sentinel-2 index measurements.' : 'CNN and ViT are RGB learned-embedding baselines.'}</p></div></div>
      {!isSpectral && <><label htmlFor="run">RGB MODEL</label><select id="run" value={runId} onChange={(event) => setRunId(event.target.value)}>{runs.map((run) => <option value={run.run_id} key={run.run_id}>{run.display_name}</option>)}</select>{selectedRun && <div className="run-details"><span>{selectedRun.searchable_gallery_count.toLocaleString()} searchable training scenes</span><span>Validation macro P@5: {(selectedRun.validation_macro_precision_at_5_mean * 100).toFixed(1)}% ± {(selectedRun.validation_macro_precision_at_5_std * 100).toFixed(1)} points across {selectedRun.validation_seed_count} seeds</span><span>Representative checkpoint: seed {selectedRun.seed} · {selectedRun.run_id}</span></div>}<p className="helper">This baseline checks scene-label similarity. It does not measure human relevance.</p></>}
      {isSpectral && <p className="helper spectral-note">RGB images are display context. The spectral score uses NDWI, MNDWI and NDVI from the aligned Sentinel-2 bands.</p>}
      {requiresBefore && <><div className="divider"/><label htmlFor="before-search">{parsed?.intent === 'spectral_change_similarity' ? 'BEFORE EXAMPLE TILE' : 'EXAMPLE TILE'}</label><input id="before-search" value={beforeSearch} onChange={(event) => findTiles(event.target.value, 'before')} placeholder="Search tile ID…"/>{beforeTiles.length > 0 && <select value={beforeTile} onChange={(event) => setBeforeTile(event.target.value)}><option value="">Choose an example tile</option>{beforeTiles.map((tile) => <option key={tile.tile_id} value={tile.tile_id}>{tile.tile_id} · {tile.date?.slice(0, 10)}</option>)}</select>}</>}
      {parsed?.intent === 'spectral_change_similarity' && <><label htmlFor="after-search">AFTER EXAMPLE TILE</label><input id="after-search" value={afterSearch} onChange={(event) => findTiles(event.target.value, 'after')} placeholder="Search later observation…"/>{afterTiles.length > 0 && <select value={afterTile} onChange={(event) => setAfterTile(event.target.value)}><option value="">Choose the later observation</option>{afterTiles.map((tile) => <option key={tile.tile_id} value={tile.tile_id}>{tile.tile_id} · {tile.date?.slice(0, 10)}</option>)}</select>}</>}
      <div className="divider"/><label htmlFor="count">RESULTS</label><select id="count" value={count} onChange={(event) => setCount(Number(event.target.value))}>{[5, 10, 20, 30].map((value) => <option key={value} value={value}>Top {value}</option>)}</select><button className="primary retrieve-button" disabled={!canRetrieve || busy === 'retrieve'} onClick={retrieve}>{busy === 'retrieve' ? 'Searching exact gallery…' : 'Retrieve evidence'} <span>→</span></button><p className="micro-note">Exact ranking · training sequences only · no generated answer</p>
    </aside></section>

    {error && <div className="error-banner"><b>Could not continue</b><span>{error}</span><button onClick={() => setError('')}>Dismiss</button></div>}
    {results && <section className="results-section"><div className="results-head"><div><div className="eyebrow">RGB RETRIEVAL · {results.retrieval_method.replaceAll('_', ' ')}</div><h2>{results.results.length ? `${results.results.length} evidence scenes` : 'No matching scenes'}</h2><p>{results.candidate_count.toLocaleString()} training-gallery candidates · {results.run_id}</p><p className="query-method">Query vector: {results.query_vector_method}</p></div><div className="result-query">“{results.query.original_query}”</div></div>{results.notice && <div className="notice">{results.notice}</div>}<div className="results-grid">{results.results.map((item) => <article className="result-card" key={item.tile_id}><div className="image-wrap"><img src={item.preview_url} alt={`Sentinel-2 RGB composite of ${item.tile_id}`} loading="lazy"/><span className="rank">{String(item.rank).padStart(2, '0')}</span><span className="score">cos {item.score.toFixed(3)}</span></div><div className="result-body"><div className="result-id">{item.tile_id}</div><div className="result-date">{item.timestamp?.slice(0, 10) ?? 'Date unavailable'}</div><div className="label-row">{item.labels.map((label) => <span className={`label-pill ${label === 'flood' ? 'flood' : 'no-flood'}`} key={label}>{label.replace('_', ' ')}</span>)}</div><div className="result-meta">{item.latitude?.toFixed(3)}°, {item.longitude?.toFixed(3)}° · {item.sensor}</div><div className="band-list">RGB display · learned RGB embedding</div></div></article>)}</div><div className="provenance">Flood/no-flood labels describe the scene/date, not flooded pixels. RGB cosine scores are not probabilities or spectral measurements.</div></section>}

    {spectralResults && <section className="results-section"><div className="results-head"><div><div className="eyebrow">SPECTRAL RETRIEVAL · {spectralResults.retrieval_method.replaceAll('_', ' ')}</div><h2>{spectralResults.results.length ? `${spectralResults.results.length} spectral evidence results` : 'No matching results'}</h2><p>{spectralResults.candidate_count.toLocaleString()} eligible scenes or date pairs · {spectralResults.similarity_measure}</p></div><div className="result-query">“{spectralResults.query.original_query}”</div></div>{spectralResults.notice && <div className="notice">{spectralResults.notice}</div>}<div className="spectral-grid">{spectralResults.results.map((item) => <article className="spectral-card" key={`${item.before_tile_id ?? ''}:${item.tile_id ?? item.after_tile_id}`}><div className="spectral-images">{item.before_preview_url && <img src={item.before_preview_url} alt="Before-date RGB preview" loading="lazy"/>}{item.preview_url && <img src={item.preview_url} alt="RGB scene preview" loading="lazy"/>}{item.after_preview_url && <img src={item.after_preview_url} alt="After-date RGB preview" loading="lazy"/>}</div><div className="spectral-maps">{item.index_url && <img src={item.index_url} alt="MNDWI spectral evidence map" loading="lazy"/>}{item.before_index_url && <img src={item.before_index_url} alt="Before-date MNDWI map" loading="lazy"/>}{item.after_index_url && <img src={item.after_index_url} alt="After-date MNDWI map" loading="lazy"/>}{item.change_map_url && <img src={item.change_map_url} alt="After-minus-before MNDWI change map" loading="lazy"/>}</div><div className="spectral-card-body"><div className="spectral-rank">{String(item.rank).padStart(2, '0')} · {item.score.toFixed(3)}</div><div className="result-id">{item.tile_id ?? `${item.before_tile_id} → ${item.after_tile_id}`}</div><div className="result-date">{item.before_date ? `${item.before_date.slice(0, 10)} → ${item.after_date?.slice(0, 10)} (${item.gap_days} days)` : item.after_date?.slice(0, 10)}</div><div className="result-meta">{item.latitude?.toFixed(3)}°, {item.longitude?.toFixed(3)}° · sequence {item.sequence_id} · {item.label?.replace('_', ' ')}</div><p className="score-description">{item.score_description}</p><div className="index-stats">{Object.entries(item.change_summary ?? item.index_summary).map(([name, value]) => <span key={name}><b>{name.toUpperCase()}</b> median {value.median.toFixed(3)} · positive {Math.round(value.positive_fraction * 100)}%</span>)}</div></div></article>)}</div><div className="provenance">RGB images are display previews. NDWI uses B03/B08, MNDWI uses B03/B11, and NDVI uses B08/B04. Positive MNDWI and changes are spectral clues, not pixel flood labels or probabilities. Stored source values are not asserted to be calibrated surface reflectance; SEN12-FLOOD labels remain scene/date-level.</div></section>}
    <footer><span>GEORAG · M11</span><span>MULTISPECTRAL INDEX EVIDENCE → EXACT RETRIEVAL</span><a href="https://github.com/madhavpr191221/georag" target="_blank" rel="noreferrer">SOURCE ↗</a></footer>
  </main>
}

export default App
