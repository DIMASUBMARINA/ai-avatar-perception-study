'use strict';
// Experiment server: assigns counterbalanced blocks, serves stimuli under opaque ids
// (ground truth never reaches the browser) and appends finished sessions to a CSV.
const express = require('express');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const PORT = +process.env.PORT || 3000;
const TPB = +process.env.TRIALS_PER_BLOCK || 20;          // trials per block, must be even; half real / half AI
const LIMIT_MS = +process.env.VIEW_LIMIT_MS || 3000;
const ROOT = __dirname;
const STIM_DIR = process.env.STIM_DIR || path.join(ROOT, 'stimuli');
const DATA_DIR = process.env.DATA_DIR || path.join(ROOT, 'data');
const OUT_FILE = path.join(DATA_DIR, 'participant_trials.csv');
const MAJORS = ['IT', 'Humanities', 'Other'];
const COLS = ['participant_id', 'condition', 'block_order', 'block_index', 'trial_index', 'stimulus_id', 'is_ai',
  'generator', 'user_choice', 'response_time_ms', 'exposure_ms', 'focus_lost', 'button_layout', 'major',
  'device_type', 'submitted_at'];
if (TPB % 2) throw new Error('TRIALS_PER_BLOCK must be even');

const walk = d => fs.existsSync(d) ? fs.readdirSync(d, { withFileTypes: true })
  .flatMap(e => e.isDirectory() ? walk(path.join(d, e.name)) : [path.join(d, e.name)]) : [];

function loadStimuli() {
  const map = new Map(), byClass = { real: [], ai: [] };
  for (const cls of ['real', 'ai']) {
    const base = path.join(STIM_DIR, cls);
    for (const f of walk(base)) {
      if (!/\.(jpe?g|png|webp|svg)$/i.test(f)) continue;
      const rel = path.relative(STIM_DIR, f).split(path.sep).join('/');
      const parts = path.relative(base, f).split(path.sep);
      const s = { id: crypto.createHash('sha1').update(rel).digest('hex').slice(0, 10), file: f,
        is_ai: cls === 'ai' ? 1 : 0, generator: parts.length > 1 ? parts[0] : '', demo: /demo/i.test(rel) };
      map.set(s.id, s); byClass[cls].push(s);
    }
  }
  return { map, byClass };
}
const stim = loadStimuli();
const isDemo = [...stim.map.values()].some(s => s.demo);

const shuffle = a => { for (let i = a.length - 1; i > 0; i--) { const j = crypto.randomInt(i + 1); [a[i], a[j]] = [a[j], a[i]]; } return a; };
const csv = v => { const s = v === null || v === undefined ? '' : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
const countParticipants = () => fs.existsSync(OUT_FILE)
  ? new Set(fs.readFileSync(OUT_FILE, 'utf8').split('\n').slice(1).filter(Boolean).map(l => l.split(',')[0])).size : 0;

fs.mkdirSync(DATA_DIR, { recursive: true });
let started = countParticipants();   // block order alternates AB / BA by arrival
const sessions = new Map();

const app = express();
app.use(express.json({ limit: '200kb' }));
app.use(express.static(path.join(ROOT, 'public')));

app.get('/api/info', (_req, res) => res.json({ demo: isDemo, limitMs: LIMIT_MS }));

app.get('/stim/:id', (req, res) => {
  const s = stim.map.get(req.params.id);
  if (!s) return res.sendStatus(404);
  res.set('Cache-Control', 'private, max-age=3600');
  res.sendFile(s.file);
});

app.post('/api/session', (req, res) => {
  const { major, device_type } = req.body || {};
  if (!MAJORS.includes(major) || !['mobile', 'pc'].includes(device_type)) return res.status(400).json({ error: 'Некорректный профиль' });
  if (stim.byClass.real.length < TPB || stim.byClass.ai.length < TPB)
    return res.status(500).json({ error: `Нужно минимум ${TPB} изображений в stimuli/real и в stimuli/ai` });
  const half = TPB / 2, real = shuffle(stim.byClass.real.slice()), ai = shuffle(stim.byClass.ai.slice());
  const order = started++ % 2 === 0 ? ['3s_limit', 'unrestricted'] : ['unrestricted', '3s_limit'];
  const blocks = order.map((condition, b) => ({   // disjoint stimuli per block, 1:1 real/AI in each
    condition,
    trials: shuffle([...real.slice(b * half, (b + 1) * half), ...ai.slice(b * half, (b + 1) * half)])
      .map(s => ({ stimulus_id: s.id, url: '/stim/' + s.id })),
  }));
  const participant_id = 'P' + crypto.randomBytes(4).toString('hex');
  const layout = crypto.randomInt(2) ? 'real_left' : 'ai_left';
  sessions.set(participant_id, { order, blocks, major, device_type, layout, done: false });
  res.json({ session_id: participant_id, limitMs: LIMIT_MS, layout, blocks });
});

app.post('/api/session/:id/finish', (req, res) => {
  const s = sessions.get(req.params.id);
  if (!s) return res.status(404).json({ error: 'Сессия не найдена' });
  if (s.done) return res.status(409).json({ error: 'Уже сохранено' });
  const resp = (req.body || {}).responses;
  const expected = s.blocks.flatMap((b, bi) => b.trials.map((t, ti) => ({ ...t, condition: b.condition, bi, ti })));
  if (!Array.isArray(resp) || resp.length !== expected.length) return res.status(400).json({ error: 'Неполные данные' });
  const byId = new Map(resp.map(r => [r.stimulus_id, r]));
  const now = new Date().toISOString();
  const rows = [];
  for (const e of expected) {
    const r = byId.get(e.stimulus_id);
    if (!r || ![0, 1].includes(r.choice) || !Number.isFinite(r.rt_ms) || r.rt_ms < 0 || r.rt_ms > 6e5) return res.status(400).json({ error: 'Некорректный ответ' });
    const truth = stim.map.get(e.stimulus_id);
    rows.push([req.params.id, e.condition, s.order[0] === '3s_limit' ? '3s_first' : 'unrestricted_first', e.bi, e.ti,
      e.stimulus_id, truth.is_ai, truth.generator, r.choice, r.rt_ms,
      Number.isFinite(r.exposure_ms) ? r.exposure_ms : '', r.focus_lost ? 1 : 0, s.layout, s.major, s.device_type, now]);
  }
  const header = fs.existsSync(OUT_FILE) ? '' : COLS.join(',') + '\n';
  fs.appendFileSync(OUT_FILE, header + rows.map(r => r.map(csv).join(',')).join('\n') + '\n');
  s.done = true;
  res.json({ ok: true, rows: rows.length });
});

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`Experiment on http://localhost:${PORT} | stimuli: real=${stim.byClass.real.length} ai=${stim.byClass.ai.length} | ${TPB} trials/block`);
    if (isDemo) console.log('WARNING: demo placeholder stimuli detected. Do NOT collect study data with them.');
    console.log(`Data -> ${OUT_FILE}`);
  });
}
module.exports = app;
