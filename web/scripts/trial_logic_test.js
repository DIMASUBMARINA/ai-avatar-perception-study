'use strict';
// Checks the trial timing logic of public/app.js with a fake DOM (no browser needed).
const fs = require('fs'), path = require('path'), vm = require('vm'), assert = require('assert');

function makeEnv() {
  const mkEl = () => {
    const L = {};
    return { hidden: false, dataset: {}, src: '', textContent: '',
      addEventListener: (t, f) => (L[t] = L[t] || new Set()).add(f),
      removeEventListener: (t, f) => L[t] && L[t].delete(f),
      fire: (t, ev) => [...(L[t] || [])].forEach(f => f(ev || {})) };
  };
  const els = {};
  const get = sel => els[sel] || (els[sel] = mkEl());
  const btnReal = get('#btnReal'), btnAi = get('#btnAi');
  btnReal.dataset.choice = '0'; btnAi.dataset.choice = '1';
  btnReal.dataset.side = 'left'; btnAi.dataset.side = 'right';
  const docL = mkEl();
  const document = { hidden: false, querySelector: get, querySelectorAll: () => [btnReal, btnAi],
    addEventListener: docL.addEventListener, removeEventListener: docL.removeEventListener, fire: docL.fire };
  const ctx = vm.createContext({ document, window: { scrollTo() {} }, performance, setTimeout, clearTimeout,
    requestAnimationFrame: cb => setTimeout(cb, 0), matchMedia: () => ({ matches: false }), fetch: async () => ({}), console });
  const src = fs.readFileSync(path.join(__dirname, '..', 'public', 'app.js'), 'utf8').replace(/\ninit\(\);\s*$/, '\n');
  vm.runInContext(src, ctx);
  vm.runInContext('session = { limitMs: 3000 };', ctx);
  return { ctx, get, btnReal, btnAi, document };
}

async function trial(cond, answerAfterMs, btn) {
  const env = makeEnv();
  env.ctx.t = { stimulus_id: 'x', url: '/stim/x' };
  const p = vm.runInContext(`runTrial(t, '${cond}', 'test')`, env.ctx);
  await new Promise(r => setTimeout(r, 600 + 50 + answerAfterMs));   // fixation + first frames + wanted delay
  const stimHiddenBefore = env.get('#stim').hidden, answersVisible = !env.get('#answers').hidden;
  env[btn].fire('click');
  const r = await p;
  return { r, stimHiddenBefore, answersVisible };
}

(async () => {
  // 1. 3s condition, early answer: face disappears with the answer, exposure == RT
  let x = await trial('3s_limit', 700, 'btnAi');
  assert.strictEqual(x.r.choice, 1);
  assert.ok(x.r.rt_ms > 600 && x.r.rt_ms < 1000, 'early rt ' + x.r.rt_ms);
  assert.ok(Math.abs(x.r.exposure_ms - x.r.rt_ms) < 1, 'exposure must equal rt on early answers');
  assert.ok(x.answersVisible && !x.stimHiddenBefore, 'answers and face must be visible together');

  // 2. 3s condition, late answer: face hidden by the timer at ~3000 ms, buttons stay, RT counted from onset
  x = await trial('3s_limit', 3600, 'btnReal');
  assert.strictEqual(x.r.choice, 0);
  assert.ok(x.stimHiddenBefore && x.answersVisible, 'after the limit the face is hidden but buttons stay');
  assert.ok(x.r.exposure_ms >= 2990 && x.r.exposure_ms < 3100, 'exposure ' + x.r.exposure_ms);
  assert.ok(x.r.rt_ms > 3500 && x.r.rt_ms < 3900, 'late rt ' + x.r.rt_ms);

  // 3. unrestricted: face stays, exposure == RT
  x = await trial('unrestricted', 1200, 'btnAi');
  assert.ok(!x.stimHiddenBefore, 'face must stay visible');
  assert.ok(Math.abs(x.r.exposure_ms - x.r.rt_ms) < 1);

  // 4. held key (auto-repeat) must not answer
  const env = makeEnv(); env.ctx.t = { stimulus_id: 'x', url: '/stim/x' };
  const p = vm.runInContext("runTrial(t, 'unrestricted', 'test')", env.ctx);
  await new Promise(r => setTimeout(r, 700));
  vm.runInContext("session.layout = 'real_left'", env.ctx);
  env.document.fire('keydown', { key: 'ArrowRight', repeat: true });
  let settled = false; p.then(() => (settled = true));
  await new Promise(r => setTimeout(r, 100));
  assert.ok(!settled, 'auto-repeated key must be ignored');
  env.document.fire('keydown', { key: 'ArrowRight', repeat: false });
  assert.strictEqual((await p).choice, 1);
  // 5. the final score is shown only when the server returned a valid one
  const sc = makeEnv(), el = sc.get('#score'); el.hidden = true;
  vm.runInContext('showScore({})', sc.ctx);
  assert.ok(el.hidden, 'no score must be shown without data');
  vm.runInContext('showScore({ correct: 31, total: 40 })', sc.ctx);
  assert.ok(!el.hidden && el.textContent.includes('31 of 40') && el.textContent.includes('78%'), el.textContent);
  // 6. review screen data: one item per trial, in the order shown, with the truth
  const rv = makeEnv();
  rv.ctx.blocks = [
    { condition: '3s_limit', trials: [{ stimulus_id: 'a', url: '/stim/a' }, { stimulus_id: 'b', url: '/stim/b' }] },
    { condition: 'unrestricted', trials: [{ stimulus_id: 'c', url: '/stim/c' }, { stimulus_id: 'd', url: '/stim/d' }] },
  ];
  rv.ctx.answers = [{ stimulus_id: 'a', choice: 1 }, { stimulus_id: 'b', choice: 1 }, { stimulus_id: 'c', choice: 0 }, { stimulus_id: 'd', choice: 1 }];
  rv.ctx.truthById = { a: 1, b: 0, c: 0, d: 0 };
  const items = vm.runInContext('buildReview(blocks, answers, truthById)', rv.ctx);
  assert.deepStrictEqual(items.map(i => [i.url, i.condition, i.correct]),
    [['/stim/a', '3s_limit', true], ['/stim/b', '3s_limit', false], ['/stim/c', 'unrestricted', true], ['/stim/d', 'unrestricted', false]]);
  rv.ctx.items = items;
  const sm = vm.runInContext('summarize(items)', rv.ctx);
  assert.deepStrictEqual(JSON.parse(JSON.stringify(sm)), {
    byCondition: { '3s_limit': { correct: 1, total: 2 }, unrestricted: { correct: 1, total: 2 } },
    total: { correct: 2, total: 4 },
    realCalledAi: { n: 2, of: 3 },     // b and d are real photos answered "AI"
    aiCalledReal: { n: 0, of: 1 },
  });
  console.log('OK: trial timing logic');
})().catch(e => { console.error('FAIL:', e.message); process.exit(1); });
