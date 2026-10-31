'use strict';
const $ = s => document.querySelector(s);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const frame = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
const VIEWS = ['consent', 'profile', 'intro', 'trial', 'done', 'results', 'error'];
const show = n => { VIEWS.forEach(v => $('#v-' + v).hidden = v !== n); window.scrollTo(0, 0); };
const post = (u, b) => fetch(u, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) });
const round1 = x => Math.round(x * 10) / 10;

let session = null, truth = null;
const responses = [];

const INTRO = {
  '3s_limit': ['Short exposure', 'The face is shown for at most 3 seconds. Answer as soon as you have decided: the face disappears right after your answer. If you do not answer in time, it disappears by itself and the buttons stay.',
    'Go with your first impression.'],
  unrestricted: ['Free viewing', 'The face stays on screen until you answer. Look for as long as you need.',
    'Accuracy matters more than speed.'],
};

async function init() {
  const info = await (await fetch('/api/info')).json();
  $('#demo').hidden = !info.demo;
  show('consent');
}
$('#agree').onchange = e => { $('#toProfile').disabled = !e.target.checked; };
$('#toProfile').onclick = () => show('profile');
const priorValue = () => { const r = document.querySelector('input[name="prior"]:checked'); return r ? r.value : ''; };
const checkProfile = () => { $('#start').disabled = !($('#major').value && priorValue()); };
$('#major').onchange = checkProfile;
document.querySelectorAll('input[name="prior"]').forEach(r => { r.onchange = checkProfile; });
$('#start').onclick = async () => {
  $('#start').disabled = true; $('#startNote').hidden = false;
  const device_type = navigator.maxTouchPoints > 0 && matchMedia('(pointer: coarse)').matches ? 'mobile' : 'pc';
  try {
    const r = await post('/api/session', { major: $('#major').value, prior_exposure: priorValue(), device_type });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.status);
    session = j;
  } catch (e) { return fail('Could not start: ' + e.message, () => $('#start').onclick()); }
  $('#startNote').hidden = true;
  await runExperiment();
};

function fail(msg, again) { $('#errText').textContent = msg; $('#retry').onclick = again; show('error'); }

function waitClick(btn) { return new Promise(r => btn.addEventListener('click', r, { once: true })); }

async function preload(trials) {
  await Promise.all(trials.map(t => { const i = new Image(); i.src = t.url; return i.decode().catch(() => {}); }));
}

function layoutButtons() {   // answer-button side is randomised per participant to avoid side bias
  const realLeft = session.layout === 'real_left';
  $('#btnReal').dataset.side = realLeft ? 'left' : 'right';
  $('#btnAi').dataset.side = realLeft ? 'right' : 'left';
}

function waitChoice() {
  return new Promise(resolve => {
    const btns = [...document.querySelectorAll('#answers button')];
    const done = c => { cleanup(); resolve(c); };
    const onKey = e => {
      if (e.repeat) return;   // a held key must not answer the next trial
      const side = e.key === 'ArrowLeft' ? 'left' : e.key === 'ArrowRight' ? 'right' : null;
      const b = side && btns.find(x => x.dataset.side === side);
      if (b) done(+b.dataset.choice);
    };
    const handlers = btns.map(b => { const h = () => done(+b.dataset.choice); b.addEventListener('click', h); return [b, h]; });
    function cleanup() { document.removeEventListener('keydown', onKey); handlers.forEach(([b, h]) => b.removeEventListener('click', h)); }
    document.addEventListener('keydown', onKey);
  });
}

async function runTrial(t, cond, label) {
  const stim = $('#stim'), answers = $('#answers'), fix = $('#fix');
  $('#progress').textContent = label;
  answers.hidden = true; stim.hidden = true; fix.hidden = false;
  await sleep(600);                                   // fixation
  let focusLost = false;
  const onVis = () => { if (document.hidden) focusLost = true; };
  document.addEventListener('visibilitychange', onVis);
  stim.src = t.url; fix.hidden = true; stim.hidden = false; answers.hidden = false;   // face and answer buttons together
  await frame();
  const t0 = performance.now();                       // stimulus painted: the response clock starts here in BOTH conditions
  let timerId = null, hiddenAt = null;
  if (cond === '3s_limit') {                          // the face disappears after the limit, buttons stay available
    timerId = setTimeout(() => { stim.hidden = true; hiddenAt = performance.now(); },
      Math.max(0, session.limitMs - (performance.now() - t0)));
  }
  const choice = await waitChoice();
  const rt = performance.now() - t0;
  clearTimeout(timerId);
  document.removeEventListener('visibilitychange', onVis);
  const exposure = hiddenAt !== null ? hiddenAt - t0 : rt;   // time the face was on screen: until the answer or until the limit
  answers.hidden = true; stim.hidden = true;
  return { stimulus_id: t.stimulus_id, choice, rt_ms: round1(rt), exposure_ms: round1(exposure), focus_lost: focusLost };
}

async function runExperiment() {
  layoutButtons();
  for (const [bi, block] of session.blocks.entries()) {
    const [title, text, hint] = INTRO[block.condition];
    $('#introTitle').textContent = `Block ${bi + 1} of ${session.blocks.length}: ${title}`;
    $('#introText').textContent = text; $('#introHint').textContent = hint;
    show('intro');
    $('#introGo').disabled = true; $('#introGo').textContent = 'Loading...';
    await preload(block.trials);
    $('#introGo').disabled = false; $('#introGo').textContent = 'Start block';
    await waitClick($('#introGo'));
    show('trial'); $('#keyHint').hidden = matchMedia('(pointer: coarse)').matches;
    for (const [ti, t] of block.trials.entries()) {
      const r = await runTrial(t, block.condition, `Block ${bi + 1} of ${session.blocks.length} · trial ${ti + 1} of ${block.trials.length}`);
      responses.push(r);
    }
  }
  await save();
}

function showScore(d) {
  const el = $('#score');
  if (!Number.isFinite(d.correct) || !(d.total > 0)) return;
  el.textContent = `Your score: ${d.correct} of ${d.total} correct (${Math.round(100 * d.correct / d.total)}%).`;
  el.hidden = false;
}

const CHOICE_TEXT = ['Real photo', 'AI-generated'];
const COND_TEXT = { '3s_limit': 'Short exposure', unrestricted: 'Free viewing' };

// One item per trial in the order shown: what was on screen, what the participant answered, what was true.
function buildReview(blocks, answers, truthById) {
  const byId = new Map(answers.map(a => [a.stimulus_id, a]));
  return blocks.flatMap(b => b.trials.map(t => {
    const choice = byId.get(t.stimulus_id).choice, is_ai = truthById[t.stimulus_id];
    return { url: t.url, condition: b.condition, choice, is_ai, correct: choice === is_ai };
  }));
}

function summarize(items) {
  const by = {};
  for (const it of items) { const c = by[it.condition] = by[it.condition] || { correct: 0, total: 0 }; c.total++; c.correct += it.correct; }
  const real = items.filter(i => i.is_ai === 0), ai = items.filter(i => i.is_ai === 1);
  return { byCondition: by, total: { correct: items.filter(i => i.correct).length, total: items.length },
    realCalledAi: { n: real.filter(i => i.choice === 1).length, of: real.length },
    aiCalledReal: { n: ai.filter(i => i.choice === 0).length, of: ai.length } };
}

function renderResults() {
  const items = buildReview(session.blocks, responses, truth), s = summarize(items);
  const lines = [`Overall: ${s.total.correct} of ${s.total.total} correct.`];
  for (const [cond, c] of Object.entries(s.byCondition)) lines.push(`${COND_TEXT[cond]}: ${c.correct} of ${c.total} correct.`);
  lines.push(`Real photos you took for AI-generated: ${s.realCalledAi.n} of ${s.realCalledAi.of}.`);
  lines.push(`AI-generated faces you took for real photos: ${s.aiCalledReal.n} of ${s.aiCalledReal.of}.`);
  const sum = $('#resSummary'); sum.textContent = '';
  for (const l of lines) { const li = document.createElement('li'); li.textContent = l; sum.appendChild(li); }
  const list = $('#review'); list.textContent = '';
  items.forEach((it, i) => {
    const li = document.createElement('li'); li.className = it.correct ? 'ok' : 'bad';
    const img = document.createElement('img'); img.src = it.url; img.alt = `Face ${i + 1}`; img.loading = 'lazy';
    const v = document.createElement('div'); v.className = 'verdict'; v.textContent = it.correct ? 'Correct' : 'Wrong';
    const d = document.createElement('div');
    d.textContent = `It was: ${CHOICE_TEXT[it.is_ai]}. You answered: ${CHOICE_TEXT[it.choice]}. (${COND_TEXT[it.condition]}, #${i + 1})`;
    li.append(img, v, d); list.appendChild(li);
  });
  applyFilter();
}

function applyFilter() {
  const only = $('#onlyMistakes').checked;
  const cards = [...$('#review').children];
  cards.forEach(li => { li.hidden = only && li.className === 'ok'; });
  $('#noMistakes').hidden = !(only && cards.length && cards.every(li => li.className === 'ok'));
}

$('#showResults').onclick = () => { renderResults(); show('results'); };
$('#backToDone').onclick = () => show('done');
$('#onlyMistakes').onchange = applyFilter;

async function save() {
  try {
    const r = await post(`/api/session/${session.session_id}/finish`, { responses });
    if (r.ok || r.status === 409) {   // 409: already stored on an earlier attempt
      const d = await r.json().catch(() => ({}));
      showScore(d);
      if (d.truth && typeof d.truth === 'object') { truth = d.truth; $('#showResults').hidden = false; }
      return show('done');
    }
    throw new Error((await r.json()).error || r.status);
  } catch (e) { fail('Your answers were not saved (' + e.message + '). Check your connection and retry.', save); }
}

init();
