/* ═══════════════════════════════════════════════════════════════
   पूछो — client
   Three jobs: drive the dot-matrix field, talk to the API, and make
   the two-tier answer legible (fast grounded answer first, LLM polish
   second — the second can only replace the first, never remove it).
   ═══════════════════════════════════════════════════════════════ */

const $  = (s) => document.querySelector(s);
const esc = (s) => (s ?? '').replace(/[&<>"]/g, (c) => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;' }[c]));
const ms = (v) => `${(+v).toFixed(1)}ms`;

const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.text()).slice(0, 240) || `HTTP ${r.status}`);
  return r.json();
};

/* ───────────────────────── dot-matrix field ─────────────────────────
   A grid of dots displaced by two travelling sine waves. `energy` is
   driven by live microphone amplitude while recording and by a decaying
   pulse while a query is in flight, so the background is a readout of
   system state rather than ambience.                                  */
(() => {
  const cv = document.getElementById('field');
  if (!cv || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const ctx = cv.getContext('2d', { alpha: true });

  let w = 0, h = 0, dpr = 1, cols = 0, rows = 0, GAP = 30;

  function size() {
    // DPR capped at 1.5: the field is soft by design, so the extra pixels cost
    // fill rate without being visible.
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    w = cv.clientWidth; h = cv.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Hold the dot count near a budget rather than a fixed pitch, so a large
    // display doesn't quadratically increase per-frame work.
    GAP = Math.max(26, Math.sqrt((w * h) / 2600));
    cols = Math.ceil(w / GAP) + 1;
    rows = Math.ceil(h / GAP) + 1;
  }
  size();
  addEventListener('resize', size, { passive: true });

  const state = { energy: 0, target: 0, t: 0 };
  window.__field = state;

  // Pointer adds a soft local swell — rewards cursor movement without noise.
  let px = -999, py = -999;
  addEventListener('pointermove', (e) => { px = e.clientX; py = e.clientY; }, { passive: true });
  addEventListener('pointerleave', () => { px = py = -999; });

  // Square dots, drawn in alpha buckets. Setting fillStyle is the expensive
  // call, so instead of ~3,000 state changes per frame we make eight: every dot
  // is binned by opacity and each bin is filled in one pass. Squares also read
  // more like an LED matrix than circles, and fillRect is far cheaper than arc.
  const BINS = 8;
  const bins = Array.from({ length: BINS }, () => []);
  let last = 0;

  function frame(now) {
    requestAnimationFrame(frame);
    if (now - last < 32) return;            // ~30fps is plenty for this motion
    last = now;

    state.t += 0.02;
    state.energy += (state.target - state.energy) * 0.06;
    state.target *= 0.985;                   // decay toward calm

    ctx.clearRect(0, 0, w, h);
    const E = state.energy;
    const hot = E > 0.1;
    for (let b = 0; b < BINS; b++) bins[b].length = 0;

    for (let i = 0; i < cols; i++) {
      const wi = i * 0.22, si = Math.sin(wi + state.t * 2.1);
      const x = i * GAP;
      for (let j = 0; j < rows; j++) {
        const y = j * GAP;
        const wave = si * Math.cos(j * 0.19 - state.t * 1.5)
                   + Math.sin((i + j) * 0.11 + state.t * 1.2);

        // vertical falloff keeps the field quiet behind the headline
        const fall = y / h * 1.5 + 0.18;

        const dx = px - x, dy = py - y;
        const d2 = dx * dx + dy * dy;
        const near = d2 < 36100 ? 1 - Math.sqrt(d2) / 190 : 0;

        const a = (0.05 + wave * 0.05 + E * 0.2) * (fall > 1 ? 1 : fall) + near * 0.34;
        if (a <= 0.02) continue;

        const s = Math.min(2.6, (1.1 + wave * 0.8) * (0.5 + E * 1.4) + near * 2.4);
        if (s <= 0.35) continue;

        const b = Math.min(BINS - 1, (a * BINS / 0.5) | 0);
        bins[b].push(x, y + wave * 9 * E, s);
      }
    }

    for (let b = 0; b < BINS; b++) {
      const arr = bins[b];
      if (!arr.length) continue;
      const a = ((b + 0.5) / BINS) * 0.5;
      ctx.fillStyle = hot
        ? `rgba(255,${(107 + 110 * (1 - Math.min(1, E))) | 0},${(53 + 160 * (1 - Math.min(1, E))) | 0},${a})`
        : `rgba(255,255,255,${a})`;
      for (let k = 0; k < arr.length; k += 3) {
        const s = arr[k + 2];
        ctx.fillRect(arr[k], arr[k + 1], s, s);
      }
    }
  }
  requestAnimationFrame(frame);
})();

const pulse = (v) => { if (window.__field) window.__field.target = Math.max(window.__field.target, v); };

/* ───────────────────────── health ───────────────────────── */
let SERVING = [];
api('/health').then((h) => {
  SERVING = h.serving || [];
  const chip = $('#chipIndex');
  chip.classList.add('ready');
  chip.querySelector('span').textContent =
    `${h.total_chunks.toLocaleString()} chunks · ${h.serving.join('+')}`;
  $('#footHost').textContent = `${h.embedder_variant} · ${h.index_tag}`;
  if (!h.stt_configured) {
    $('#micBtn').disabled = true;
    $('#hint').textContent = 'Voice disabled — no STT key on this server. Typing works.';
  }
}).catch(() => {
  $('#chipIndex').querySelector('span').textContent = 'server unreachable';
});

/* ───────────────────────── answer rendering ───────────────────────── */
const shell = $('#answerShell');

function setTier(el, state, value) {
  el.dataset.state = state;
  if (value !== undefined) el.querySelector('em').textContent = value;
}

function renderBudget(msValue) {
  const pctRaw = (msValue / 200) * 100;
  const fill = $('#budgetFill');
  fill.style.width = `${Math.min(100, pctRaw)}%`;
  fill.classList.toggle('over', msValue > 200);
  $('#budgetLabel').innerHTML = msValue > 200
    ? `<span style="color:var(--refuse)">${ms(msValue)} — over budget</span>`
    : `${ms(msValue)} · ${(100 - pctRaw).toFixed(0)}% of budget unused`;
}

function renderAnswer(d, tier) {
  shell.hidden = false;
  document.body.classList.add('answered');   // hero compacts, answer takes the stage

  const ans = $('#answer');
  ans.textContent = d.answer || '(no answer)';
  ans.classList.toggle('muted', ['abstain', 'refusal', 'greeting'].includes(d.answer_source));

  // tier track
  const t1 = document.querySelector('.tier.t1');
  const t2 = document.querySelector('.tier.t2');
  setTier(t1, 'active', ms(d.fast_path_ms));
  // "generated" alone is ambiguous when the model returns the extracted span
  // verbatim -- which it does whenever that span is already a complete answer.
  // Distinguishing rewritten from unchanged shows the LLM had an effect (or
  // honestly reports that it had none) instead of leaving the viewer guessing.
  const rewritten = !!d.generated_answer && d.generated_answer !== d.extractive_answer;
  if (tier === 'generated') {
    setTier(t2, 'active', `${ms(d.total_ms)} · ${rewritten ? 'rewritten' : 'unchanged'}`);
  } else if (tier === 'pending') {
    setTier(t2, 'pending', '···');
  } else if (d.reason === 'llm_reported_insufficient') {
    // The LLM ran and judged the context inadequate. Saying "declined" rather
    // than showing a blank makes it clear generation happened and had an
    // opinion -- which is the point of running it at all.
    setTier(t2, 'declined', 'declined');
  } else if (d.llm_error) {
    setTier(t2, 'declined', 'failed');
  } else {
    setTier(t2, 'idle', '—');
  }

  renderBudget(d.fast_path_ms);

  // verdicts — the guardrail decisions, stated plainly
  const v = [];
  const src = d.answer_source;
  if (src === 'refusal')       v.push(`<span class="v bad">refused · ${esc(d.reason || 'unsafe intent')}</span>`);
  else if (src === 'greeting') v.push(`<span class="v">not a question · no retrieval spent</span>`);
  else if (src === 'abstain')  v.push(`<span class="v warn">abstained · ${esc(d.reason === 'llm_reported_insufficient' ? 'model judged context inadequate' : 'not supported by corpus')}</span>`);
  else                         v.push(`<span class="v good">grounded</span>`);
  if (d.support   != null) v.push(`<span class="v">support ${d.support.toFixed(3)}</span>`);
  if (d.grounding != null) v.push(`<span class="v">grounding ${d.grounding.toFixed(3)}</span>`);
  if (d.citations?.length) v.push(`<span class="v good">cited [${d.citations.join(', ')}]</span>`);
  if (src === 'generated') {
    v.push(rewritten
      ? `<span class="v good">LLM rewrote the span</span>`
      : `<span class="v">LLM returned the span verbatim — nothing to improve</span>`);
  }
  if (d.stt_ms)   v.push(`<span class="v">STT ${ms(d.stt_ms)} · outside budget</span>`);
  if (d.llm_error) v.push(`<span class="v bad">LLM ${esc(d.llm_error.slice(0, 48))}</span>`);
  $('#verdicts').innerHTML = v.join('');

  // Model-knowledge answer, only ever alongside an abstain and never merged
  // into the answer above it.
  const un = $('#unsourced');
  if (un) {
    un.innerHTML = d.unsourced_answer
      ? `<div class="unsourced">
           <div class="tag-un">⚠ not from the corpus · model's own knowledge</div>
           <p lang="hi">${esc(d.unsourced_answer)}</p>
           <div class="caveat">unverified — no source, no citation, not covered by the grounding check</div>
         </div>`
      : '';
  }

  const before = $('#beforeAfter');
  if (before) {
    before.innerHTML = rewritten
      ? `<details><summary>what the LLM changed ↘</summary>
           <div class="src"><b>01 extracted</b><br>${esc(d.extractive_answer)}</div>
           <div class="src" style="border-color:var(--grounded)"><b>02 generated</b><br>${esc(d.generated_answer)}</div>
         </details>`
      : '';
  }

  $('#sources').innerHTML = d.sources?.length
    ? `<details><summary>${d.sources.length} retrieved passages ↘</summary>` +
      d.sources.map((s, i) => `
        <div class="src">[${i + 1}] ${esc(s.text.slice(0, 320))}
          <div class="meta">${esc(s.unit_id)} · rrf ${s.score}${
            Object.keys(s.contributors || {}).length ? ' · via ' + esc(Object.keys(s.contributors).join(', ')) : ''
          }</div>
        </div>`).join('') + `</details>`
    : '';

  shell.scrollIntoView({ block: 'center', behavior: 'smooth' });
}

/* ───────────────────────── ask ───────────────────────── */
let busy = false;

async function ask(question) {
  question = (question || '').trim();
  if (!question || busy) return;
  busy = true;
  pulse(0.9);
  $('#hint').classList.remove('err');
  $('#hint').textContent = 'retrieving…';

  try {
    // Tier 1 — extractive. This is the number the 200ms budget is measured against.
    const fast = await api('/ask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, generate: false }),
    });
    renderAnswer(fast, fast.decision === 'allow' ? 'pending' : 'idle');
    $('#hint').textContent = `answered in ${ms(fast.fast_path_ms)} — no LLM involved`;
    pulse(0.5);

    // Tier 2 — LLM polish. Can only replace the answer, never remove it.
    if (fast.decision === 'allow') {
      try {
        const full = await api('/ask', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, generate: true }),
        });
        renderAnswer(full, full.answer_source === 'generated' ? 'generated' : 'idle');
        $('#hint').textContent = full.answer_source === 'generated'
          ? `polished in ${ms(full.total_ms)} · fast answer stood at ${ms(full.fast_path_ms)}`
          : `kept the extracted answer — ${esc(full.reason || 'generation not used')}`;
      } catch {
        setTier(document.querySelector('.tier.t2'), 'idle', '—');
        $('#hint').textContent = 'generation unavailable — the grounded answer stands';
      }
    }
  } catch (e) {
    $('#hint').classList.add('err');
    $('#hint').textContent = e.message;
  } finally {
    busy = false;
  }
}

$('#askBtn').onclick = () => ask($('#q').value);
$('#q').addEventListener('keydown', (e) => { if (e.key === 'Enter') ask($('#q').value); });
$('#closeAns').onclick = () => {
  shell.hidden = true;
  document.body.classList.remove('answered');
};
document.querySelectorAll('.sample').forEach((b) => {
  b.onclick = () => { $('#q').value = b.dataset.q; ask(b.dataset.q); };
});

/* ───────────────────────── microphone ─────────────────────────
   Sarvam accepts wav/mp3; MediaRecorder emits webm/opus, so the blob is
   decoded and re-encoded to 16kHz mono PCM in the browser. Doing it here
   keeps ffmpeg off the server and matches Sarvam's recommended input.  */
let recorder = null, chunks = [], analyser = null, audioCtx = null, meterRAF = 0;

function encodeWav(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); str(36, 'data'); v.setUint32(40, samples.length * 2, true);
  let o = 44;
  for (let i = 0; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

async function toWav(blob) {
  const ac = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await ac.decodeAudioData(await blob.arrayBuffer());
  const rate = 16000;
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * rate), rate);
  const src = off.createBufferSource();
  src.buffer = decoded; src.connect(off.destination); src.start();
  const out = await off.startRendering();
  ac.close();
  return encodeWav(out.getChannelData(0), rate);
}

function meter() {
  if (!analyser) return;
  const buf = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(buf);
  let peak = 0;
  for (let i = 0; i < buf.length; i++) peak = Math.max(peak, Math.abs(buf[i] - 128) / 128);
  if (window.__field) window.__field.target = Math.min(1.4, peak * 3.2);
  meterRAF = requestAnimationFrame(meter);
}

$('#micBtn').onclick = async () => {
  const btn = $('#micBtn');
  if (recorder && recorder.state === 'recording') { recorder.stop(); return; }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];

    // live amplitude drives the background field
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    audioCtx.createMediaStreamSource(stream).connect(analyser);
    meter();

    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => chunks.push(e.data);

    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      cancelAnimationFrame(meterRAF); analyser = null;
      audioCtx?.close(); audioCtx = null;
      btn.classList.remove('rec');
      $('#hint').textContent = 'transcribing…';

      try {
        const wav = await toWav(new Blob(chunks, { type: chunks[0]?.type || 'audio/webm' }));
        const fd = new FormData();
        fd.append('audio', wav, 'question.wav');
        fd.append('generate', 'true');
        const d = await api('/voice', { method: 'POST', body: fd });

        if (!d.stt_ok || !d.transcript) {
          $('#hint').classList.add('err');
          $('#hint').textContent = `couldn't hear that — ${esc(d.stt_error || 'empty transcript')}`;
          return;
        }
        $('#q').value = d.transcript;
        renderAnswer(d, d.answer_source === 'generated' ? 'generated' : 'idle');
        $('#hint').textContent =
          `heard “${d.transcript}” · STT ${ms(d.stt_ms)} (outside budget) · answer ${ms(d.fast_path_ms)}`;
      } catch (e) {
        $('#hint').classList.add('err');
        $('#hint').textContent = e.message;
      }
    };

    recorder.start();
    btn.classList.add('rec');
    $('#hint').classList.remove('err');
    $('#hint').textContent = 'listening — click again to stop (30s max)';
    setTimeout(() => { if (recorder?.state === 'recording') recorder.stop(); }, 30000);
  } catch (e) {
    $('#hint').classList.add('err');
    $('#hint').textContent = `microphone blocked — ${esc(e.message)}`;
  }
};

/* ───────────────────────── live benchmark ───────────────────────── */
function countTo(el, target, decimals = 0, dur = 1100) {
  const from = parseFloat(el.textContent) || 0;
  const t0 = performance.now();
  const set = (v) => { el.textContent = v.toFixed(decimals); };
  let settled = false;

  const step = (now) => {
    if (settled) return;
    const p = Math.min(1, (now - t0) / dur);
    set(from + (target - from) * (1 - Math.pow(1 - p, 3)));
    if (p < 1) requestAnimationFrame(step); else settled = true;
  };
  requestAnimationFrame(step);

  // rAF is throttled in background tabs, which would leave the tile showing a
  // stale placeholder while the API reported something else. The animation is
  // decoration; the value is not, so guarantee it lands either way.
  setTimeout(() => { if (!settled) { settled = true; set(target); } }, dur + 150);
}

$('#benchBtn').onclick = async () => {
  const btn = $('#benchBtn');
  if (btn.classList.contains('busy')) return;
  btn.classList.add('busy');
  btn.querySelector('span').textContent = 'running 100…';
  pulse(1.0);

  try {
    const d = await api('/benchmark?n=100');
    const p = d.fast_path_ms;
    countTo($('#mP50'), p.p50, 1);
    countTo($('#mP70'), p.p70, 1);
    countTo($('#mP100'), p.p100, 1);
    $('#mHit').textContent = `${d.within_budget}/${d.n_queries}`;
    btn.querySelector('span').textContent = `${d.n_queries} queries · live`;

    $('#stageRows').innerHTML = Object.entries(d.stages_ms).map(([k, s]) =>
      `<tr><td>${esc(k)}</td><td>${s.p50}</td><td>${s.p70}</td><td>${s.p90}</td><td>${s.p99}</td><td>${s.p100}</td></tr>`
    ).join('') +
      `<tr class="total"><td>fast path total</td><td>${p.p50}</td><td>${p.p70}</td><td>${p.p90}</td><td>${p.p99}</td><td>${p.p100}</td></tr>`;
  } catch (e) {
    btn.querySelector('span').textContent = 'failed — retry';
  } finally {
    btn.classList.remove('busy');
  }
};

/* ───────────────────────── strategy comparison ───────────────────────── */
$('#cmpBtn').onclick = async () => {
  const question = $('#q').value.trim();
  const btn = $('#cmpBtn');
  if (!question) { btn.textContent = 'ask something first ↑'; return; }
  btn.textContent = 'querying every index…';
  pulse(0.8);

  try {
    const d = await api('/compare', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    btn.textContent = `re-run · ${esc(d.agreement)}`;
    $('#compare').innerHTML = `
      <table class="grid-table">
        <thead><tr><th>Strategy</th><th>Chunks</th><th>Search</th><th>Extract</th><th>Support</th></tr></thead>
        <tbody>${d.configs.map((c) => `
          <tr class="${c.is_served ? 'served' : ''}">
            <td>${esc(c.config)}${c.is_served ? '<span class="tag">served</span>' : ''}</td>
            <td>${c.chunks.toLocaleString()}</td>
            <td>${c.search_ms}</td>
            <td>${c.extract_ms}</td>
            <td>${c.support}</td>
          </tr>`).join('')}
        </tbody>
      </table>`;
  } catch (e) {
    btn.textContent = 'comparison failed — retry';
  }
};
