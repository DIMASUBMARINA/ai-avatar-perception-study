'use strict';
const $ = s => document.querySelector(s);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const frame = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
const VIEWS = ['consent', 'profile', 'intro', 'trial', 'done', 'error'];
const show = n => { VIEWS.forEach(v => $('#v-' + v).hidden = v !== n); window.scrollTo(0, 0); };
const post = (u, b) => fetch(u, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) });
const round1 = x => Math.round(x * 10) / 10;

let session = null;
const responses = [];

const INTRO = {
  '3s_limit': ['Короткий показ', 'Лицо покажется не дольше чем на 3 секунды. Отвечайте, как только решите: после ответа лицо сразу исчезнет. Если не успеете, оно исчезнет само, а кнопки останутся.',
    'Отвечайте по первому впечатлению.'],
  unrestricted: ['Свободный просмотр', 'Лицо останется на экране, пока вы не ответите. Смотрите столько, сколько нужно.',
    'Точность важнее скорости.'],
};

async function init() {
  const info = await (await fetch('/api/info')).json();
  $('#demo').hidden = !info.demo;
  show('consent');
}
$('#agree').onchange = e => { $('#toProfile').disabled = !e.target.checked; };
$('#toProfile').onclick = () => show('profile');
$('#major').onchange = () => { $('#start').disabled = !$('#major').value; };
$('#start').onclick = async () => {
  $('#start').disabled = true; $('#startNote').hidden = false;
  const device_type = navigator.maxTouchPoints > 0 && matchMedia('(pointer: coarse)').matches ? 'mobile' : 'pc';
  try {
    const r = await post('/api/session', { major: $('#major').value, device_type });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.status);
    session = j;
  } catch (e) { return fail('Не удалось начать: ' + e.message, () => $('#start').onclick()); }
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
    $('#introTitle').textContent = `Блок ${bi + 1} из ${session.blocks.length}: ${title}`;
    $('#introText').textContent = text; $('#introHint').textContent = hint;
    show('intro');
    $('#introGo').disabled = true; $('#introGo').textContent = 'Загрузка...';
    await preload(block.trials);
    $('#introGo').disabled = false; $('#introGo').textContent = 'Начать блок';
    await waitClick($('#introGo'));
    show('trial'); $('#keyHint').hidden = matchMedia('(pointer: coarse)').matches;
    for (const [ti, t] of block.trials.entries()) {
      const r = await runTrial(t, block.condition, `Блок ${bi + 1} из ${session.blocks.length} · проба ${ti + 1} из ${block.trials.length}`);
      responses.push(r);
    }
  }
  await save();
}

async function save() {
  try {
    const r = await post(`/api/session/${session.session_id}/finish`, { responses });
    if (r.ok || r.status === 409) return show('done');   // 409: already stored on an earlier attempt
    throw new Error((await r.json()).error || r.status);
  } catch (e) { fail('Ответы не сохранились (' + e.message + '). Проверьте соединение и повторите.', save); }
}

init();
