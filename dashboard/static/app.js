// Silent Voice live dashboard: draws each snapshot from dashboard/server.py (dashboard/state.py) as Thai's Figma v5.
// Text frames are snapshots (JSON); binary frames are cam.preview JPEGs with their face points (server.py pack_frame).
'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const SRC = { camera: 'camera', emg: 'emg', claude: 'claude', lips: 'camera', muscles: 'emg' };
const TONE_COLOR = { happy: '--warm', sad: '--concerned', neutral: '--calm', surprised: '--surprised', angry: '--firm' };
// Pad = landmark + a * (nose-to-chin axis) + b * (cheek-to-cheek axis); defaults measured on session 2, C recalibrates.
const PAD_DEFAULT = {
  A0: { lm: 152, a: 0.1092, b: -0.0342, left: false }, A1: { lm: 379, a: -0.0212, b: -0.0043, left: false },
  A2: { lm: 150, a: -0.0298, b: 0.021, left: true }, A3: { lm: 411, a: 0.0126, b: -0.0134, left: false },
};
const AXIS = { top: 168, chin: 152, left: 234, right: 454 };
const PAD_ORDER = ['A0', 'A1', 'A2', 'A3'];
const MOUTH = { left: 61, right: 291, top: 13, bottom: 14 };
const W = 760, H = 1080, PREVIEW_STALE_MS = 2000;
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

let snap = null, lastPreview = -Infinity, speaking = false, toneLabel = 'neutral';
let frame = null, framePending = false, lastFace = null;

// ---- the 1920 x 1080 stage, scaled to the window -------------------------------------------------------
function fit() {
  const k = Math.min(innerWidth / 1920, innerHeight / 1080);
  $('stage').style.transform = `translate(-50%, -50%) scale(${k})`;
}
addEventListener('resize', fit); fit();

// ---- connection ------------------------------------------------------------------------------------------
function connect() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = e => {
    if (typeof e.data === 'string') { snap = JSON.parse(e.data); render(snap); return; }
    onFrame(e.data);
  };
  ws.onclose = () => setTimeout(connect, 1000);
}
// ?offline: no websocket; a test hands snapshots to window.reclaimRender (dashboard/demo_feed.py --fixtures)
window.reclaimRender = s => { snap = s; render(s); };
if (!new URLSearchParams(location.search).has('offline')) connect();

// ---- the Call / Text switch ------------------------------------------------------------------------------
// The call server owns the mode; a click (or M) asks the dashboard server to switch it. The thumb moves at once
// and settles on what the call server says; if it can't switch, the thumb goes back and says why.
let modeWant = null, noteTimer = null;
function renderMode(s) {
  const el = $('mode'), ch = modeWant || s.channel || 'call';
  el.dataset.ch = ch;
  el.classList.toggle('off', !s.switchable);
  el.classList.toggle('busy', !!modeWant);
  el.querySelectorAll('button').forEach(b => {
    b.setAttribute('aria-checked', String(b.dataset.ch === ch));
    b.disabled = !s.switchable;
  });
}
async function switchMode(ch) {
  if (!snap || !snap.switchable || modeWant || ch === snap.channel) return;
  modeWant = ch;
  renderMode(snap);
  try {
    const r = await fetch('/api/channel', { method: 'POST', body: JSON.stringify({ channel: ch }),
      headers: { 'Content-Type': 'application/json', 'X-Silent-Voice': '1' } });
    if (!r.ok) throw new Error((await r.text()) || r.statusText);
    snap.channel = (await r.json()).channel;
    modeNote('');
  } catch (e) {
    modeNote(`Couldn't switch: ${e.message}`);
  } finally {
    modeWant = null;
    renderMode(snap);
  }
}
function modeNote(text) {
  const el = $('mode-note');
  el.textContent = text;
  el.classList.toggle('show', !!text);
  clearTimeout(noteTimer);
  if (text) noteTimer = setTimeout(() => el.classList.remove('show'), 5000);
}
$('mode').addEventListener('click', e => { const b = e.target.closest('button'); if (b) switchMode(b.dataset.ch); });
addEventListener('keydown', e => {
  if ((e.key === 'm' || e.key === 'M') && !e.ctrlKey && !e.metaKey && !e.altKey && snap && snap.mode !== 'end')
    switchMode(snap.channel === 'sms' ? 'call' : 'sms');
});

// ---- live screen -----------------------------------------------------------------------------------------
const mmss = s => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

function render(s) {
  $('live').classList.toggle('hidden', s.mode === 'end');
  $('end').classList.toggle('hidden', s.mode !== 'end');
  if (s.mode === 'end') { renderEnd(s); return; }
  const live = s.mode === 'live';
  $('live-dot').classList.toggle('off', !live);
  const sms = s.channel === 'sms';
  $('title').textContent = !live ? 'Waiting for the call' : sms ? 'Live call, the judge is texting' : 'Live call with the judge';
  $('judge-via').textContent = sms ? '· texted from the phone on the call' : '· transcribed by Cartesia Ink';
  $('sub').textContent = live ? `${mmss(s.elapsed)}${s.number ? ' on ' + s.number : ''}` : (s.number || '');
  $('m-lips').textContent = s.metrics.lips_to_voice == null ? '--' : (s.metrics.lips_to_voice / 1000).toFixed(2) + ' s';
  $('m-fixed').textContent = s.metrics.fixed;
  $('m-repeats').textContent = s.metrics.repeats;
  renderMode(s);
  renderConversation(s);
  renderLatency(s);
  $('status').classList.toggle('warn', !s.status.ok);
  $('status').querySelector('span').textContent = s.status.text;
  renderCamera(s);
}

// ---- the conversation ------------------------------------------------------------------------------------
// Reads top to bottom in time, anchored to the bottom like a chat: older lines (small) fade out at the top, then
// the judge's latest line, then Austin's earlier sentences in this answer (small), then the sentence he is
// mouthing now (big). Nothing still part of the exchange is ever dropped.
//
// Motion: every bubble is a keyed element that lives across snapshots. When the layout changes, each one glides
// from where it was drawn to its new place (measure, change, then animate the difference: FLIP), a bubble that
// grows reveals its new line as the column moves up, Austin's finished sentence shrinks into the conversation, and
// each new word rises in. A change mid-motion starts from where things are on screen, so motion never jumps.
const MOTION = !matchMedia('(prefers-reduced-motion: reduce)').matches;
const EASE = 'cubic-bezier(.22, 1, .36, 1)', MOVE_MS = 460, ENTER_MS = 380;
let convSig = null, flips = new Map();          // element -> {anim, clipR, clipB}: the motion running on it

function renderConversation(s) {
  const t = s.transcript;
  const lastJudge = s.judge_live ? -1 : t.map(x => x.who).lastIndexOf('judge');
  const judge = s.judge_live || (lastJudge >= 0 ? t[lastJudge].text : '');
  const a = s.austin;
  // a finished sentence stays big until the judge answers (or starts to)
  const answered = !!s.judge_live || (lastJudge >= 0 && t[lastJudge].t > (a && a.t || 0));
  const current = a && a.words.length && !(a.final && answered) ? a : null;
  // the transcript line that is the current sentence (a restarted decoder reuses utt_id: match the newest after it began)
  const curLine = current ? t.filter(x => x.who === 'austin' && x.utt_id === current.utt_id && x.t >= (current.t || 0)).pop() : null;
  const small = x => ({ key: 'l' + x.seq, cls: 'past ' + x.who, text: x.text });
  const keep = (x, i) => i !== lastJudge && x !== curLine;
  const before = t.filter((x, i) => (lastJudge < 0 || i < lastJudge) && keep(x, i)).map(small);
  const turn = lastJudge < 0 ? [] : t.filter((x, i) => i > lastJudge && keep(x, i)).map(small);
  // what the big bubbles hold: when it changes, the old content leaves as the small bubble with that key
  const judgeId = s.judge_live ? 'live' : lastJudge >= 0 ? 'l' + t[lastJudge].seq : '';
  const austinId = current ? 'u' + current.utt_id + '@' + current.t : 'empty-' + s.mode;   // the same as it finishes
  const words = current ? current.words : null;
  const panel = explainShape(s, current);
  const sig = JSON.stringify([s.mode, judge, judgeId, austinId, before.map(x => x.key + x.text), turn.map(x => x.key),
    words && words.map(w => [w.w, w.was]), panel]);
  if (sig === convSig) {                // nothing moved: only the live values (bars, colours) update, in place
    updateWords(words, current);
    renderExplain(s, current, false);
    return;
  }

  // FIRST: where everything is drawn now (mid-motion included), keyed by what it holds
  const first = new Map(), mid = $('middle'), jb = $('judge-text'), ab = $('austin-bubble');
  const k = $('stage').getBoundingClientRect().width / 1920 || 1;
  const seen = el => { const r = el.getBoundingClientRect(), f = flips.get(el);
    return r.width ? { x: r.left, y: r.top, w: r.width - (f ? f.clipR * k * progressLeft(f) : 0),
                       h: r.height - (f ? f.clipB * k * progressLeft(f) : 0) } : null; };
  mid.querySelectorAll('[data-key]').forEach(el => first.set(el.dataset.key, seen(el)));
  const prevJudge = jb.dataset.id, prevAustin = ab.dataset.id;
  // the small bubble Austin's previous big sentence becomes, once it has left the big bubble
  const was = ab.dataset.utt ? t.filter(x => x.who === 'austin' && String(x.utt_id) === ab.dataset.utt &&
                                             x.t >= parseFloat(ab.dataset.start)).pop() : null;
  const leftKey = was && prevAustin !== austinId ? 'l' + was.seq : null;
  if (prevJudge) first.set('big:' + prevJudge, seen(jb));
  if (prevAustin) first.set('big:' + prevAustin, seen(ab));
  for (const el of [$('judge-label'), $('austin-label'), $('explain')]) first.set('#' + el.id, seen(el));
  for (const f of flips.values()) f.anim.cancel();
  flips.clear();

  // CHANGE: the new content, keyed elements reused
  $('judge').classList.toggle('hidden', !judge);
  if (jb.textContent !== judge) jb.textContent = judge;
  jb.classList.toggle('live', !!s.judge_live);
  jb.dataset.id = judgeId;
  $('austin').classList.toggle('waiting', s.mode !== 'live');
  ab.dataset.id = austinId;
  ab.dataset.utt = current ? current.utt_id : '';
  ab.dataset.start = current ? current.t || 0 : '';
  reconcile($('history'), before);
  reconcile($('turn'), turn);
  const fresh = prevAustin !== austinId;
  if (fresh) $('words').replaceChildren();
  if (current) updateWords(words, current); else showEmpty(s.mode === 'live' ? 'Listening' : 'Not on a call yet');
  renderExplain(s, current, false);
  const judges = t.filter(x => x.who === 'judge'), prev = judges[judges.length - (s.judge_live ? 1 : 2)];
  const aw = getComputedStyle($('austin')).getPropertyValue('--aw');
  fitConversation(prev ? prev.seq : 0);
  convSig = sig;                        // only once the change is fully on screen: an error retries next snapshot

  // LAST + PLAY: each element glides from where it was to where it is now
  if (!MOTION) return;
  const play = (el, from, opts = {}) => {
    const to = el.getBoundingClientRect();
    if (!to.width) return;
    if (!from) {                        // new on screen: rise in
      el.animate([{ opacity: 0, transform: 'translateY(18px)' }, { opacity: 1, transform: 'none' }],
                 { duration: ENTER_MS, easing: EASE, delay: opts.delay || 0, fill: 'backwards' });
      return;
    }
    let dx = (from.x - to.left) / k, dy = (from.y - to.top) / k, scale = 1;
    if (opts.morph) {                   // a big bubble becoming this small one: shrink into place
      scale = Math.min(4, Math.max(0.25, from.h / to.height));
      const right = el.classList.contains('austin');
      dx = ((right ? from.x + from.w - to.width * scale : from.x) - to.left) / k;
      dy = (from.y + from.h - to.height * scale - to.top) / k;
    }
    const clipR = opts.grow ? Math.max(0, (to.width - from.w) / k) : 0;
    const clipB = opts.grow ? Math.max(0, (to.height - from.h) / k) : 0;
    if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5 && scale === 1 && !clipR && !clipB) return;
    const radius = getComputedStyle(el).borderRadius || '0';
    const frames = [{ transform: `translate(${dx}px, ${dy}px) scale(${scale})`, transformOrigin: '0 0' },
                    { transform: 'none', transformOrigin: '0 0' }];
    if (clipR || clipB) {               // grew: the new part is revealed as it slides into place
      frames[0].clipPath = `inset(0 ${clipR}px ${clipB}px 0 round ${radius})`;
      frames[1].clipPath = `inset(0 0 0 0 round ${radius})`;
    }
    const anim = el.animate(frames, { duration: MOVE_MS, easing: EASE });
    flips.set(el, { anim, clipR, clipB });
    anim.onfinish = anim.oncancel = () => { if (flips.get(el) && flips.get(el).anim === anim) flips.delete(el); };
  };
  mid.querySelectorAll('[data-key]').forEach(el => {
    if (el.classList.contains('hidden')) return;
    const key = el.dataset.key, own = first.get(key);
    if (own) play(el, own);
    else if (key === leftKey && first.get('big:' + prevAustin)) play(el, first.get('big:' + prevAustin), { morph: true });
    else if (first.get('big:' + key)) play(el, first.get('big:' + key), { morph: true });
    else play(el, null);
  });
  play($('judge-label'), first.get('#judge-label'));
  play($('austin-label'), first.get('#austin-label'));
  play($('explain'), first.get('#explain'));
  // the big bubbles: the same content glides and grows; new content rises in
  play(jb, prevJudge === judgeId || (prevJudge === 'live' && judgeId.startsWith('l')) ? first.get('big:' + prevJudge) : null, { grow: true });
  play(ab, fresh ? null : first.get('big:' + prevAustin), { grow: true });
  if (!fresh && aw && aw !== getComputedStyle($('austin')).getPropertyValue('--aw')) {   // the text stepped down a size
    const ratio = parseFloat(aw) / parseFloat(getComputedStyle($('austin')).getPropertyValue('--aw'));
    $('words').animate([{ transform: `scale(${ratio})`, transformOrigin: '0 0' }, { transform: 'none', transformOrigin: '0 0' }],
                       { duration: MOVE_MS, easing: EASE });
  }
}

function progressLeft(f) {              // how much of a running motion is still ahead (0 to 1), eased
  const p = f.anim.effect.getComputedTiming().progress;
  return p == null ? 0 : 1 - p;
}

// keyed children: reuse each bubble across snapshots, so it can move instead of being redrawn
function reconcile(box, items) {
  const old = new Map([...box.children].map(el => [el.dataset.key, el]));
  items.forEach((it, i) => {
    let el = old.get(it.key);
    if (!el) { el = document.createElement('div'); el.dataset.key = it.key; }
    old.delete(it.key);
    el.className = it.cls;
    if (el.textContent !== it.text) el.textContent = it.text;
    if (box.children[i] !== el) box.insertBefore(el, box.children[i] || null);
  });
  old.forEach(el => el.remove());
}

// Austin's words, one element per word kept across snapshots: a new word rises in, a word that locks turns from
// grey to black and its bar fills (CSS transitions), a word Claude repaired shows the old one struck out above it
function updateWords(words, current) {
  if (!words) return;
  const box = $('words'), thr = current.cands && current.cands.threshold;
  box.classList.remove('empty');
  if (box.firstElementChild && box.firstElementChild.dataset.empty) box.replaceChildren();
  words.forEach((w, i) => {
    let el = box.children[i];
    if (!el) {
      el = document.createElement('div');
      el.innerHTML = '<span class="was"></span><span class="w"></span><div class="bar"><i></i></div>';
      if (MOTION) el.classList.add('enter');
      box.appendChild(el);
    }
    el.classList.toggle('word', true);
    el.classList.toggle('pending', !w.locked);
    const [was, ww, bar] = [el.children[0], el.children[1], el.children[2].firstElementChild];
    if (ww.textContent !== w.w) ww.textContent = w.w;
    was.textContent = w.was || '';
    was.hidden = !w.was;
    bar.className = 'src-' + (w.locked ? (SRC[w.src] || 'camera') : 'pending');
    bar.style.width = barWidth(w, thr) + '%';
  });
  while (box.children.length > words.length) box.lastElementChild.remove();
}

function showEmpty(text) {
  const box = $('words');
  box.classList.add('empty');
  if (!(box.firstElementChild && box.firstElementChild.dataset.empty === text)) {
    box.innerHTML = `<div class="word${MOTION ? ' enter' : ''}" data-empty="${esc(text)}"><span class="w">${esc(text)}</span></div>`;
  }
}

// Fit the exchange in the space between the header and the latency strip, the most important part last to give
// way: the text sizes step down and the candidate panel drops its note, then its rows; only then do Austin's
// earlier sentences in this answer thin to the newest, and at the smallest size the two big bubbles keep their
// newest lines (the start fades at the top). Within one exchange (one judge line and Austin's answer) the level
// only goes up, so the text never grows back while Austin mouths.
const FIT = [   // austin word px, judge px, earlier sentences kept, candidate panel
  [80, 40, 9, 'full'], [68, 40, 9, 'full'], [68, 34, 9, 'full'], [58, 34, 9, 'full'], [58, 34, 9, 'compact'],
  [50, 30, 9, 'compact'], [50, 30, 9, 'none'], [44, 28, 2, 'none'], [44, 28, 1, 'none']];
let fitKey = null, fitLevel = 0;
function fitConversation(key) {
  if (key !== fitKey) { fitKey = key; fitLevel = 0; }
  const mid = $('middle'), aus = $('austin'), jud = $('judge'), turn = $('turn'), ex = $('explain');
  const words = $('words'), jt = $('judge-text');
  const apply = ([aw, jw, keep, panel]) => {
    aus.style.setProperty('--aw', aw + 'px');
    jud.style.setProperty('--jw', jw + 'px');
    [...turn.children].forEach((el, i, all) => el.classList.toggle('hidden', i < all.length - keep));
    ex.dataset.panel = panel;
  };
  const need = () => [jud, turn, aus, ex].reduce((h, el) => h + (el.classList.contains('hidden') ? 0 : el.offsetHeight), 0);
  words.style.maxHeight = jt.style.maxHeight = '';
  let level = fitLevel;
  apply(FIT[level]);
  while (level < FIT.length - 1 && need() > mid.clientHeight) apply(FIT[++level]);
  fitLevel = level;
  let over = need() - mid.clientHeight;
  // still too tall at the smallest size: the judge's line keeps its last 4 lines, then Austin's its newest lines
  clip(jt, over > 0 ? 4 * parseFloat(getComputedStyle(jt).lineHeight) + 38 : null);
  over = need() - mid.clientHeight;
  clip(words, over > 0 ? Math.max(words.offsetHeight - over, 60) : null);
  // older lines that don't fit whole fade out at the top (no half-cut bubble)
  const hist = $('history'), top = hist.getBoundingClientRect().top - 0.5;
  [...hist.children].forEach(el => el.classList.toggle('gone', el.getBoundingClientRect().top < top));
}

function clip(el, max) {                // show the end of a too-long block, the cut-off start faded
  el.classList.toggle('clipped', max != null);
  el.style.maxHeight = max == null ? '' : max + 'px';
  if (max != null) el.scrollTop = el.scrollHeight;
}

// the bar under a word: full once locked; a pending word fills toward the lock threshold
const barWidth = (w, thr) => w.locked ? 100 : w.conf == null ? 0 : Math.round(100 * Math.min(1, Math.max(0, w.conf / (thr || 1))));

// What the panel under the caption says, without the live numbers: when only the numbers change, the bars glide
// in place (CSS transitions) instead of the panel being redrawn.
function explainShape(s, a) {
  return explainParts(s, a).text;
}

function explainParts(s, a) {
  let text = '';
  const repair = a && a.repairs.length ? a.repairs[a.repairs.length - 1] : null;
  if (a && a.repeat) {
    const w = a.repeat.words, alts = (a.repeat.alts[0] || []).slice(0, 1);
    const reader = { camera: 'Lips', emg: 'Muscles', claude: 'Claude' }[a.repeat.reader] || 'The readers';
    text += `<div class="repeat-card"><span class="ic">!</span><b>Please repeat</b><p>${esc(reader)} read ` +
            `“${esc(w[0] || '')}”${alts.length ? `, the others lean to “${esc(alts[0])}”` : ''}. ` +
            `Too close to call, so nothing was spoken.</p></div>`;
  } else if (repair) {
    text += `<div class="note claude">✦ <span>Claude changed “${esc(repair.frm)}” to “${esc(repair.to)}”: ` +
            `they look the same on the lips, and “${esc(repair.to)}” fits the conversation.</span></div>`;
  }
  const rows = a && a.cands && a.cands.rows.length ? a.cands.rows : [];
  if (rows.length) {
    const best = k => [...rows].sort((x, y) => y[k] - x[k])[0];
    const byCam = best('cam'), byEmg = best('emg'), top = rows[0];
    if (!a.repeat && !repair) {
      text += `<div class="note">${byCam.w !== top.w ? `Lips alone see “${esc(byCam.w)}”. With the conversation, “${esc(top.w)}” leads.`
        : !s.muscles ? `The lips read “${esc(top.w)}”.`
        : byEmg.w !== top.w ? `Muscles alone lean to “${esc(byEmg.w)}”. Lips and the conversation pick “${esc(top.w)}”.`
        : `Lips and muscles agree on “${esc(top.w)}”.`}</div>`;
    }
    text += `<div class="legend"><span><i class="src-camera"></i>Lips</span><span><i class="src-emg"></i>Muscles</span>` +
            `<span><i class="src-claude"></i>Claude</span><span><i class="thr"></i>Lock threshold</span></div>`;
    text += rows.map(() => '<div class="cand"><span class="cw"></span><div class="track"><i class="src-camera"></i>' +
                           '<i class="src-emg"></i><i class="src-claude"></i><span class="thr"></span></div>' +
                           '<span class="cv"></span></div>').join('');
  }
  return { text, rows, thr: a && a.cands ? a.cands.threshold : null };
}

function renderExplain(s, a, rebuild) {
  const ex = $('explain'), { text, rows, thr } = explainParts(s, a);
  if (rebuild || ex.dataset.shape !== text) { ex.innerHTML = text; ex.dataset.shape = text; }
  ex.querySelectorAll('.cand').forEach((el, i) => {
    const r = rows[i], [cam, emg, cl, mark] = el.querySelector('.track').children;
    el.classList.toggle('top', i === 0);
    el.querySelector('.cw').textContent = r.w;
    cam.style.width = 100 * r.cam + '%';
    emg.style.width = 100 * r.emg + '%';
    cl.style.width = 100 * r.claude + '%';
    mark.hidden = thr == null;
    if (thr != null) mark.style.left = 100 * thr + '%';
    el.querySelector('.cv').textContent = (r.cam + r.emg + r.claude).toFixed(2);
  });
}

function renderLatency(s) {
  const a = s.austin;
  const active = a && a.repairs.length && !a.final ? 'claude' : a && a.spoken && !a.final ? 'sonic' : 'read';
  for (const st of s.latency) {
    const el = document.querySelector(`.stage[data-k="${st.key}"]`);
    el.classList.toggle('active', st.key === active);
    el.querySelector('.sl').textContent = st.label;
    el.querySelector('.sv b').textContent = st.ms == null ? '--' : st.ms;
  }
}

// ---- camera side -----------------------------------------------------------------------------------------
function coverMap(size) {           // processed-frame normalized coords -> 760 x 1080 panel (object-fit: cover)
  const [fw, fh] = size && size[0] ? size : [1280, 720];
  const k = Math.max(W / fw, H / fh), ox = (W - fw * k) / 2, oy = (H - fh * k) / 2;
  return (x, y) => [ox + x * fw * k, oy + y * fh * k];
}

// ---- face filter: the picture and the pads on it, drawn in the same animation frame -------------------------
let frameSeq = 0;
async function onFrame(buf) {
  const seq = ++frameSeq;
  try {
    const dv = new DataView(buf);
    if (dv.getUint32(0) !== 0x52434631) return;           // 'RCF1'
    const n = dv.getUint32(4), head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
    const bitmap = await createImageBitmap(new Blob([new Uint8Array(buf, 8 + n)], { type: 'image/jpeg' }));
    if (frame && frame.seq > seq) { bitmap.close(); return; }
    if (frame) frame.bitmap.close();
    frame = { bitmap, seq, ...head };
    lastPreview = performance.now();
    if (!framePending) { framePending = true; requestAnimationFrame(drawFrame); }
  } catch (err) { /* skip a bad frame */ }
}

function drawFrame() {
  framePending = false;
  if (!frame) return;
  const f = frame, size = [f.w || f.bitmap.width, f.h || f.bitmap.height];
  const k = Math.max(W / size[0], H / size[1]);
  const g = $('mesh').getContext('2d');
  g.drawImage(f.bitmap, (W - size[0] * k) / 2, (H - size[1] * k) / 2, size[0] * k, size[1] * k);
  if (f.pts) lastFace = { pts: f.pts, size };
  placeOverlays(lastFace);
}

function faceOnPanel(face) {
  const map = coverMap(face.size), P = face.pts.map(([x, y]) => map(x, y));
  const U = [P[AXIS.chin][0] - P[AXIS.top][0], P[AXIS.chin][1] - P[AXIS.top][1]];
  const V = [P[AXIS.right][0] - P[AXIS.left][0], P[AXIS.right][1] - P[AXIS.left][1]];
  return { P, U, V };
}

function placeOverlays(face) {
  const ring = $('mouth-ring'), pads = [...$('pads').children];
  if (!face || !snap || !snap.face_ok) {
    ring.classList.add('hidden');
    layoutCallouts(pads, PAD_ORDER.map(pin => PAD_FALLBACK[pin]), 270, 490);
    return;
  }
  const { P, U, V } = faceOnPanel(face);
  const [lx, ly] = P[MOUTH.left], [rx, ry] = P[MOUTH.right], [, ty] = P[MOUTH.top], [, by] = P[MOUTH.bottom];
  const cx = (lx + rx) / 2, cy = (ty + by) / 2, w = Math.hypot(rx - lx, ry - ly) * 1.9, h = w * 82 / 183;
  Object.assign(ring.style, { width: w + 'px', height: h + 'px', left: cx - w / 2 + 'px', top: cy - h / 2 + 'px',
                              transform: `rotate(${Math.atan2(ry - ly, rx - lx)}rad)` });
  ring.classList.remove('hidden');
  const dots = PAD_ORDER.map(pin => {
    const at = padAt[pin], p = P[at.lm] || P[0];
    return [p[0] + at.a * U[0] + at.b * V[0], p[1] + at.a * U[1] + at.b * V[1]];
  });
  layoutCallouts(pads, dots, Math.min(P[AXIS.left][0], P[AXIS.right][0]), Math.max(P[AXIS.left][0], P[AXIS.right][0]));
}

const PAD_FALLBACK = { A0: [378, 672], A1: [489, 616], A2: [270, 616], A3: [489, 516] };   // no face yet

const GAP = 28, ROW = 40, LABEL_MAX_Y = 820;          // LABEL_MAX_Y keeps clear of "Heard on the call"
function layoutCallouts(pads, dots, faceLeft, faceRight) {
  for (const left of [true, false]) {
    const idx = dots.map((d, i) => i).filter(i => pads[i] && (padAt[PAD_ORDER[i]].left !== false) === left)
      .sort((a, b) => dots[a][1] - dots[b][1]);
    let prev = -Infinity;
    const ys = idx.map(i => (prev = Math.max(dots[i][1], prev + ROW)));
    const over = ys.length ? Math.max(0, ys[ys.length - 1] - LABEL_MAX_Y) : 0;
    idx.forEach((i, j) => {
      const el = pads[i], pill = el.querySelector('.pill'), [x, y] = dots[i], ly = ys[j] - over;
      const w = pill.offsetWidth || 300;
      const lx = left ? Math.max(8, faceLeft - GAP - w) : Math.min(W - 8 - w, faceRight + GAP);
      Object.assign(el.querySelector('.pdot').style, { left: `${x - 12}px`, top: `${y - 12}px` });
      Object.assign(pill.style, { left: `${lx}px`, top: `${ly - 15.5}px` });
      const ex = left ? lx + w : lx, line = el.querySelector('line');
      line.setAttribute('x1', x); line.setAttribute('y1', y); line.setAttribute('x2', ex); line.setAttribute('y2', ly);
    });
  }
}

// ---- pad calibration: press C, then click A0 to A3 -------------------------------------------------------
let padAt = loadPads(), calib = null;
function loadPads() {
  try { return { ...PAD_DEFAULT, ...JSON.parse(localStorage.getItem('reclaim.pads') || '{}') }; }
  catch { return { ...PAD_DEFAULT }; }
}
function savePads() {
  try { localStorage.setItem('reclaim.pads', JSON.stringify(padAt)); } catch { /* kept for this session only */ }
}
const PAD_PLACE = { A0: 'under his chin', A1: 'his left chin corner', A2: 'his right cheek below the mouth',
                    A3: 'his left cheek above the mouth' };

function calibHint() {
  const el = $('calib');
  el.classList.toggle('hidden', !calib);
  if (calib) el.textContent = `Click pad ${PAD_ORDER[calib.i]} (${PAD_PLACE[PAD_ORDER[calib.i]]}). Esc cancels.`;
}
addEventListener('keydown', e => {
  if (e.key === 'c' || e.key === 'C') {
    if (e.shiftKey) { padAt = { ...PAD_DEFAULT }; savePads(); calib = null; }
    else calib = lastFace ? { i: 0, next: { ...padAt } } : null;
    calibHint();
  } else if (e.key === 'Escape' && calib) { calib = null; calibHint(); }
});
$('right').addEventListener('click', e => {
  if (!calib || !lastFace) return;
  const r = $('right').getBoundingClientRect(), k = r.width / W;
  const cx = (e.clientX - r.left) / k, cy = (e.clientY - r.top) / k;
  const { P, U, V } = faceOnPanel(lastFace);
  let lm = 0;
  P.forEach((p, i) => { if (Math.hypot(p[0] - cx, p[1] - cy) < Math.hypot(P[lm][0] - cx, P[lm][1] - cy)) lm = i; });
  const dx = cx - P[lm][0], dy = cy - P[lm][1], det = U[0] * V[1] - U[1] * V[0];     // solve dx,dy = a U + b V
  if (Math.abs(det) < 1e-6) return;
  const mid = (P[AXIS.left][0] + P[AXIS.right][0]) / 2;
  calib.next[PAD_ORDER[calib.i]] = { lm, a: (dx * V[1] - dy * V[0]) / det, b: (U[0] * dy - U[1] * dx) / det,
                                     left: cx < mid };
  if (++calib.i === PAD_ORDER.length) { padAt = calib.next; savePads(); calib = null; }
  calibHint();
});

function renderCamera(s) {
  const previewOn = performance.now() - lastPreview < PREVIEW_STALE_MS;
  const face = s.face;
  const mesh = $('mesh'), g = mesh.getContext('2d');
  if (!previewOn) {                  // no video: draw the cam.face mesh
    g.clearRect(0, 0, W, H);
    if (face) {
      const map = coverMap(face.size);
      g.fillStyle = 'rgba(201,209,217,.55)';
      for (const [x, y] of face.pts) { const [px, py] = map(x, y); g.fillRect(px - 1, py - 1, 2, 2); }
      lastFace = face;
    }
  }
  $('cam-empty').classList.toggle('hidden', previewOn || !!face);
  $('cam-empty').textContent = s.status.text.startsWith('Camera') ? 'Camera: no signal' : 'Waiting for the camera';

  const chip = $('chip');
  const label = !s.face_ok ? 'Face hidden' : s.emotion.chip;    // muscle sensors off: no indicator at all
  chip.querySelector('span').textContent = label;
  chip.querySelector('i').style.background = !s.face_ok ? '#8b95a1' : css(TONE_COLOR[s.emotion.label] || '--calm');
  $('lips-off').classList.toggle('hidden', s.face_ok);

  const box = $('pads');             // content per snapshot, position per video frame
  box.classList.toggle('hidden', !s.muscles);    // sensors off: no pads drawn, rather than greyed-out ones
  if (box.children.length !== s.pads.length) {
    box.innerHTML = s.pads.map(() => `<div class="pad"><svg width="${W}" height="${H}"><line/></svg>` +
      '<span class="pdot"></span><span class="pill"><span class="pn"></span>' +
      `<span class="segs">${'<i></i>'.repeat(20)}</span></span></div>`).join('');
  }
  s.pads.forEach((p, i) => {
    const el = box.children[i], on = s.muscles && p.contact, lit = Math.round(p.level * 20);
    el.classList.toggle('on', on && p.level > 0.15);
    el.classList.toggle('off', !on);
    el.querySelector('.pn').textContent = p.name;
    [...el.querySelectorAll('.segs i')].forEach((seg, j) => seg.classList.toggle('lit', on && j < lit));
  });
  if (!previewOn) placeOverlays(face && s.face_ok ? face : lastFace);

  const a = s.austin;
  const heard = a && a.spoken ? a.spoken : '';
  speaking = !!(a && a.speaking);           // Sonic's first audio arrived and Twilio hasn't confirmed it played yet
  toneLabel = (a && a.emotion) || s.emotion.label;   // the emotion the sentence went out with
  fitHeard($('heard'), heard);
  $('tone').textContent = heard ? (a.tone || '') : '';
  $('tone').style.color = css(TONE_COLOR[toneLabel] || '--calm');
}

// What the judge heard, two lines at most: a long sentence steps down in size, then keeps its newest words
// ("…" in front), since the end is what Austin is saying now.
function fitHeard(el, text) {
  if (el.dataset.text === (text || '')) return;     // measured once per new text, not 15 times a second
  el.dataset.text = text || '';
  el.classList.remove('small');
  el.textContent = text ? `“${text}”` : '';
  if (!text || el.scrollHeight <= el.clientHeight + 1) return;
  el.classList.add('small');
  const ws = text.split(' ');
  for (let i = 0; el.scrollHeight > el.clientHeight + 1 && i < ws.length - 1; i++) el.textContent = `“…${ws.slice(i + 1).join(' ')}”`;
}

// a "voice playing" indicator, not a level meter (the call's audio isn't on the bus): it moves from Sonic's first
// audio until Twilio says the sentence was heard (timing.voice), flat otherwise
function wave(t) {
  const c = $('wave'), g = c.getContext('2d'), n = 64, active = speaking;
  g.clearRect(0, 0, c.width, c.height);
  g.fillStyle = css(TONE_COLOR[toneLabel] || '--calm');
  for (let i = 0; i < n; i++) {
    const env = Math.sin(Math.PI * (i + 0.5) / n);
    const h = active ? 8 + 100 * env * (0.35 + 0.65 * Math.abs(Math.sin(t / 140 + i * 0.7) * Math.sin(t / 410 + i * 1.3))) : 8;
    g.fillRect(i * (c.width / n) + 4, (c.height - h) / 2, 8, h);
  }
  requestAnimationFrame(wave);
}
requestAnimationFrame(wave);

// ---- end of call ----------------------------------------------------------------------------------------
const PLACEHOLDER_WER = { lips: 22, lips_muscles: 14, claude: 4 };    // design values until a session is scored
const TONES = { warm: 'Warm', concerned: 'Concerned', calm: 'Calm', surprised: 'Surprised', firm: 'Firm' };
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const use = (id, cls = '') => `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true"><use href="#${id}"/></svg>`;
const liveIcon = k => document.querySelector(`.stage[data-k="${k}"] .ib svg`)?.outerHTML || '';
const ARCHIE = '<img src="/static/icons/cartesia-archie.svg" alt="">';
let endKey = null;

function renderEnd(s) {
  const m = s.summary, key = JSON.stringify(m);
  if (key === endKey) return;                 // the summary never changes once the call is over: build once
  endKey = key;
  const L = m.latency, num = v => v == null ? '--' : v;
  $('end-judge-who').textContent = s.channel === 'sms' ? 'Judge, by text' : 'Judge, heard by Cartesia Ink';
  $('end-call').textContent = `Call ended ${mmss(m.seconds)}${s.number ? ' on ' + s.number : ''}`;
  const [n, unit] = m.seconds < 120 ? [m.seconds, 'second'] : [Math.round(m.seconds / 60), 'minute'];
  $('end-head').innerHTML = `Austin held a <span class="num">${n}</span>-${unit} phone call without making a sound.`;

  const r = m.repairs[0];
  const stats = [
    ['s-mouthed', liveIcon('read'), 'Words mouthed', String(m.words), '', 'Mouthed silently, spoken in his cloned voice.'],
    ['s-lips', ARCHIE, 'Lips to voice', L.lips_to_voice == null ? '--' : (L.lips_to_voice / 1000).toFixed(2), 's',
     `Read ${num(L.read)}, lock ${num(L.lock)}, Claude ${num(L.claude)}, voice ${num(L.sonic)} ms.`],
    ['s-claude', use('claude-mark'), 'Fixed by Claude', String(m.repairs.length), '',
     !r ? 'No look-alikes needed fixing.' : m.repairs.length === 1 ? `“${r.frm}” to “${r.to}”, a lip look-alike.`
       : `${m.repairs.length} lip look-alikes, like “${r.frm}” to “${r.to}”.`],
    ['s-repeat', use('repeat-icon'), 'Asked to repeat', String(m.repeats.length), '',
     m.repeats.length ? 'Readers disagreed, so nothing was spoken.' : 'Every word was clear enough to speak.'],
  ];
  $('end-stats').innerHTML = stats.map(([cls, icon, label, value, unit, tip]) =>
    `<div class="stat ${cls}"><span class="wm-clip"><span class="wm">${cls === 's-lips' ? liveIcon('sonic') : icon}</span></span><div class="tip">${esc(tip)}</div>` +
    `<div class="top"><span class="ic">${icon}</span><span class="k">${label}</span></div>` +
    `<b class="num">${esc(value)}${unit && value !== '--' ? `<small>${unit}</small>` : ''}</b></div>`).join('');

  const b = m.benchmark, wer = b || PLACEHOLDER_WER;
  const lab = (b && b.labels) || {};
  const rows = [[lab.lips || 'Lips alone', wer.lips, '--camera'], [lab.lips_muscles || '+ muscles', wer.lips_muscles, '--emg'],
    [lab.claude || '+ Claude', wer.claude, '--claude']];
  const max = Math.max(...rows.map(x => x[1]), 1) * 1.14;
  $('end-ph').classList.toggle('hidden', !!b);
  $('end-wer-sub').textContent = `${b && b.session ? b.session : 'Session 3'}, scored against the script he mouthed.`;
  $('end-steps').classList.toggle('placeholder', !b);
  $('end-steps').innerHTML = rows.map(([name, v, col], i) => {
    const drop = i ? Math.round(rows[i - 1][1] - v) : 0, shown = b ? `${+v.toFixed(1)}%` : `[${v}%]`;
    return `<div class="step-row"><span class="sn">${name}</span><div class="st"><i style="--w:${(100 * v / max).toFixed(1)}%;` +
      `background:var(${col})"></i></div><b>${shown}</b><span class="drop">${drop > 0 ? drop + ' fewer' : ''}</span></div>`;
  }).join('');
  const pct = v => v == null ? '--' : v + '%';
  $('end-call-line').textContent = `This call: lips alone disagreed ${pct(m.disagree.lips)} of the time, muscles ${pct(m.disagree.muscles)}.`;

  renderTones(m.tones);
  $('end-latency').innerHTML = [['read', liveIcon('read'), 'Lips and muscles', L.read], ['lock', liveIcon('lock'), 'Locking', L.lock],
    ['claude', use('claude-mark'), L.claude_label, L.claude], ['sonic', ARCHIE, 'Cartesia Sonic', L.sonic]].map(([k, icon, label, ms]) =>
    `<div class="stage"><div class="ib${k === 'claude' ? ' claude' : ''}">${icon}</div><div><div class="sl">${esc(label)}</div>` +
    `<div class="sv"><b>${num(ms)}</b><span>ms</span></div></div></div>`).join('');

  renderThread(m.transcript);
  document.fonts.ready.then(() => { if (endKey === key) renderThread(m.transcript); });   // re-trim with final font metrics
  document.querySelectorAll('#end .num').forEach((el, k) => roll(el, k));
  setTimeout(() => document.querySelectorAll('#end .st i').forEach(x => x.classList.add('go')), reduceMotion ? 0 : 700);
}

// numbers roll up from 0 like an odometer; each lower digit spins one more loop before it lands
function roll(el, k) {
  const node = el.firstChild, text = node.textContent, h = parseFloat(getComputedStyle(el).lineHeight);
  const odo = document.createElement('span');
  odo.className = 'odo'; odo.style.setProperty('--h', h + 'px'); odo.setAttribute('aria-hidden', 'true');
  let n = 0;
  for (const c of text) {
    const cell = document.createElement('span');
    if (!/\d/.test(c)) { cell.textContent = c; odo.appendChild(cell); continue; }
    const loops = n++, target = 10 * loops + Number(c), col = document.createElement('span');
    col.className = 'col';
    col.innerHTML = Array.from({ length: target + 1 }, (_, i) => `<i>${i % 10}</i>`).join('');
    cell.appendChild(col); odo.appendChild(cell);
    const go = () => { col.style.transition = reduceMotion ? 'none' : `transform ${1.5 + 0.3 * loops}s cubic-bezier(.15, .7, .2, 1)`;
                       col.style.transform = `translateY(-${target * h}px)`; };
    reduceMotion ? go() : setTimeout(() => requestAnimationFrame(go), 500 + 160 * k);
  }
  const sr = document.createElement('span'); sr.className = 'sr'; sr.textContent = text;
  el.replaceChild(odo, node); el.insertBefore(sr, odo);
}


function renderTones(tones) {
  const R = 56, C = 2 * Math.PI * R, entries = Object.entries(tones || {}), total = entries.reduce((a, [, c]) => a + c, 0);
  let off = 0;
  $('end-donut').innerHTML = total ? entries.map(([t, c]) => {
    const len = C * c / total, seg = `<circle cx="72" cy="72" r="${R}" class="seg" data-name="${t}" ` +
      `style="stroke:var(--tone-${t});stroke-dasharray:${Math.max(0, len - 4).toFixed(1)} ${C.toFixed(1)};stroke-dashoffset:${(-off).toFixed(1)}"/>`;
    off += len; return seg;
  }).join('') : `<circle cx="72" cy="72" r="${R}" class="seg empty"/>`;
  $('end-legend').innerHTML = total ? entries.map(([t, c]) =>
    `<div class="lg" data-name="${t}"><i style="background:var(--tone-${t})"></i><span>${TONES[t] || t}</span><b>${c}</b></div>`).join('')
    : '<p class="lg-empty">No tone data on this call.</p>';
  const donut = document.querySelector('#end .donut'), legend = $('end-legend');
  const show = name => {
    const seg = name && donut.querySelector(`.seg[data-name="${name}"]`);
    donut.classList.toggle('hot', !!seg);
    legend.classList.toggle('hot', !!seg);
    donut.querySelectorAll('.seg').forEach(x => x.classList.toggle('on', x === seg));
    legend.querySelectorAll('.lg').forEach(x => x.classList.toggle('on', !!seg && x.dataset.name === name));
  };
  document.querySelectorAll('#end .seg, #end .lg').forEach(el => {
    el.addEventListener('mouseenter', () => show(el.dataset.name));
    el.addEventListener('mouseleave', () => show(null));
  });
  show(null);
}

// the call as a chat; a long call keeps its newest messages and counts the rest
function renderThread(turns) {
  const bare = w => w.toLowerCase().replace(/[.,!?;:"“”]/g, '');
  const msgs = turns.map(x => {
    if (x.who === 'judge') return `<div class="msg judge"><div class="bubble">${esc(x.text)}</div></div>`;
    const ws = x.text.split(' ').map(esc);
    for (const r of x.repairs || [])
      if (r.index < ws.length && bare(ws[r.index]) === bare(r.to)) ws[r.index] = `<s>${esc(r.frm)}</s> ${ws[r.index]}`;
    const notes = [
      (x.repairs || []).length ? `<div class="cnote claude">${use('claude-mark')}Claude fixed a look-alike</div>` : '',
      x.caption_fixed ? `<div class="cnote claude">${use('claude-mark')}Claude corrected the caption (read as “${esc(x.original || '')}”)</div>` : '',
      x.repeat ? `<div class="cnote repeat">${use('repeat-icon')}Asked to repeat, nothing wrong was said</div>` : '',
      x.face_hidden ? `<div class="cnote face">${liveIcon('read')}Face hidden, muscles carried it</div>` : ''].join('');
    return `<div class="msg austin${x.repeat ? ' unspoken' : ''}"><div class="bubble">${ws.join(' ')}${x.repeat ? ' …' : ''}</div>${notes}</div>`;
  });
  const thread = $('end-thread');
  let from = 0;
  const fill = () => {
    thread.innerHTML = `<span class="sep">${from ? `${from} earlier message${from === 1 ? '' : 's'}` : 'Call started'}</span>` +
      msgs.slice(from).join('') + '<span class="sep">Call ended</span>';
  };
  fill();
  while (from < msgs.length - 1 && thread.scrollHeight > thread.clientHeight + 1) { from++; fill(); }
}
