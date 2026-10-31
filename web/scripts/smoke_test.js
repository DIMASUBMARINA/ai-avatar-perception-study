'use strict';
// End-to-end check: starts the server on a temp dir, plays 3 sessions, validates the CSV and the review data.
const { spawn, execFileSync } = require('child_process');
const fs = require('fs'), os = require('os'), path = require('path'), assert = require('assert');
const root = path.join(__dirname, '..');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'exp-'));
const PORT = 3111, base = `http://localhost:${PORT}`;
const env = { ...process.env, PORT, DATA_DIR: tmp };
if (!fs.existsSync(path.join(root, 'stimuli/real')) || !fs.readdirSync(path.join(root, 'stimuli/real')).length)
  execFileSync('node', [path.join(__dirname, 'make_demo_stimuli.js')]);

// a CSV written by an older version (different columns) must stop the server instead of being appended to
const old = fs.mkdtempSync(path.join(os.tmpdir(), 'exp-old-'));
fs.writeFileSync(path.join(old, 'participant_trials.csv'), 'participant_id,condition\nP1,3s_limit\n');
try {
  execFileSync('node', [path.join(root, 'server.js')], { env: { ...env, PORT: PORT + 1, DATA_DIR: old }, stdio: 'pipe', timeout: 10000 });
  assert.fail('the server must refuse to start on an old-format CSV');
} catch (e) {
  assert.ok(e.status && String(e.stderr).includes('different column layout'), 'wrong failure: ' + (e.stderr || e.message));
}

const srv = spawn('node', [path.join(root, 'server.js')], { env, stdio: 'ignore' });
const post = (u, b) => fetch(base + u, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) });
(async () => {
  try {
    for (let i = 0; i < 40; i++) { try { await fetch(base + '/api/info'); break; } catch { await new Promise(r => setTimeout(r, 150)); } }
    const tpb = +(process.env.TRIALS_PER_BLOCK || 20), priors = ['no', 'unsure', 'yes'], scores = {}, truths = {}, expectedPrior = {};

    // the profile must contain the prior-exposure answer
    assert.strictEqual((await post('/api/session', { major: 'IT', device_type: 'pc' })).status, 400);
    assert.strictEqual((await post('/api/session', { major: 'IT', device_type: 'pc', prior_exposure: 'maybe' })).status, 400);

    for (let k = 0; k < 3; k++) {
      const s = await (await post('/api/session', { major: 'IT', device_type: 'pc', prior_exposure: priors[k] })).json();
      expectedPrior[s.session_id] = priors[k];
      assert.strictEqual(s.blocks.length, 2);
      assert.ok(!JSON.stringify(s).includes('is_ai'), 'ground truth leaked to client');
      const responses = s.blocks.flatMap(b => b.trials.map((t, i) => ({ stimulus_id: t.stimulus_id, choice: i % 2, rt_ms: 800 + i, exposure_ms: b.condition === '3s_limit' ? 3010 : 900 + i, focus_lost: false })));
      const r = await post(`/api/session/${s.session_id}/finish`, { responses });
      assert.strictEqual(r.status, 200);
      const body = await r.json();
      assert.strictEqual(body.total, 2 * tpb);
      assert.ok(Number.isInteger(body.correct) && body.correct >= 0 && body.correct <= body.total);
      // review data: the truth of every shown stimulus, and nothing else
      assert.strictEqual(Object.keys(body.truth).length, 2 * tpb);
      assert.ok(Object.values(body.truth).every(v => v === 0 || v === 1));
      assert.deepStrictEqual(Object.keys(body.truth).sort(), responses.map(x => x.stimulus_id).sort());
      scores[s.session_id] = body.correct; truths[s.session_id] = body.truth;
      const again = await post(`/api/session/${s.session_id}/finish`, { responses });
      assert.strictEqual(again.status, 409);
      const ab = await again.json();
      assert.strictEqual(ab.correct, body.correct, 'a repeated finish must return the same score');
      assert.deepStrictEqual(ab.truth, body.truth, 'a repeated finish must return the same review data');
    }
    const lines = fs.readFileSync(path.join(tmp, 'participant_trials.csv'), 'utf8').trim().split('\n');
    const head = lines[0].split(','), rows = lines.slice(1).map(l => Object.fromEntries(l.split(',').map((v, i) => [head[i], v])));
    assert.ok(head.includes('prior_exposure'));
    assert.strictEqual(rows.length, 3 * 2 * tpb);
    for (const cond of ['3s_limit', 'unrestricted']) {
      const c = rows.filter(r => r.condition === cond);
      assert.strictEqual(c.filter(r => r.is_ai === '1').length, c.length / 2, 'classes must be 1:1');
    }
    assert.deepStrictEqual([...new Set(rows.map(r => r.block_order))].sort(), ['3s_first', 'unrestricted_first']);
    const ids = new Set(rows.filter(r => r.participant_id === rows[0].participant_id).map(r => r.stimulus_id));
    assert.strictEqual(ids.size, 2 * tpb, 'no stimulus repeated within a participant');
    for (const [pid, correct] of Object.entries(scores)) {
      const mine = rows.filter(r => r.participant_id === pid);
      assert.strictEqual(mine.filter(r => r.is_ai === r.user_choice).length, correct, 'score must match the saved rows');
      assert.ok(mine.every(r => r.prior_exposure === expectedPrior[pid]), 'prior_exposure must be saved');
      assert.ok(mine.every(r => +r.is_ai === truths[pid][r.stimulus_id]), 'review data must match the saved truth');
    }
    console.log(`OK: ${rows.length} rows, columns: ${head.join(', ')}`);
  } catch (e) { console.error('FAIL:', e.message); process.exitCode = 1; }
  finally { srv.kill(); }
})();
