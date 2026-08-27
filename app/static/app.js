'use strict';

const HUE = ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7', '#e34948'];

const COLOUR = {};
const DASH = {};
const CAP = {};
for (let p = 1; p <= 12; p++) {
  COLOUR[p] = HUE[(p - 1) % 6];
  DASH[p] = p > 6 ? '5 2.5' : 'none';
  CAP[p] = p > 6 ? 'square' : 'round';
}

function chip(p) {
  return '<u style="border-top-color:' + COLOUR[p] + ';border-top-style:'
    + (p > 6 ? 'dashed' : 'solid') + '"></u>';
}

const S = {
  engine: 'onur', folder: null, id: null, q: '',
  pads: [1], primary: 1,
  records: [], record: null,
  raw: {}, det: {}, series: {},
  view: {}, drag: null, hover: -1, big: null, busy: 0,
  mode: 'zoom', keepBase: false, markMode: false,
  axes: {}, cursor: {}, marks: {}, viewY: {},
  browse: null, pickPath: null
};

const $ = (s, r) => (r || document).querySelector(s);
const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
const el = (t, a) => {
  const n = document.createElementNS('http://www.w3.org/2000/svg', t);
  for (const k in (a || {})) { n.setAttribute(k, a[k]); }
  return n;
};
const clip = (v, a, b) => (v < a ? a : v > b ? b : v);
const fmt = (v) => (Math.abs(v) >= 1000 ? (v / 1000).toFixed(2) + 'k'
  : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

let toastTimer = null;
function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.add('on');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('on'), 3200);
}

function busy(on) {
  S.busy = Math.max(0, S.busy + (on ? 1 : -1));
  const b = $('#busy');
  if (!b.dataset.job) { b.textContent = S.busy ? 'reading…' : ''; }
}

async function api(path) {
  busy(true);
  try {
    const r = await fetch(path);
    const j = await r.json();
    if (j.error) { toast(j.error); }
    return j;
  } catch (e) {
    toast(String(e));
    return { error: String(e) };
  } finally { busy(false); }
}

/* ---------------------------------------------------------------- geometry */

function fitBox(svg) {
  const r = svg.getBoundingClientRect();
  const w = Math.max(60, Math.round(r.width));
  const h = Math.max(40, Math.round(r.height));
  const want = '0 0 ' + w + ' ' + h;
  if (svg.getAttribute('viewBox') !== want) { svg.setAttribute('viewBox', want); }
  const sel = $('.sel', svg);
  if (sel) { sel.setAttribute('height', h); }
  return { w, h };
}

function view(name, full) {
  const v = S.view[name];
  if (!v) { return full.slice(); }
  const lo = clip(v[0], full[0], full[1]);
  const hi = clip(v[1], full[0], full[1]);
  return hi - lo > (full[1] - full[0]) * 1e-6 ? [lo, hi] : full.slice();
}

function setView(name, lo, hi, full) {
  const min = (full[1] - full[0]) * 0.0004;
  if (hi - lo < min) { return; }
  S.view[name] = [Math.max(full[0], lo), Math.min(full[1], hi)];
}

function zoomAt(name, factor, at, full) {
  const v = view(name, full);
  const t = v[0] + (v[1] - v[0]) * at;
  const w = (v[1] - v[0]) * factor;
  let lo = t - w * at, hi = t + w * (1 - at);
  if (hi - lo >= full[1] - full[0]) { delete S.view[name]; return; }
  if (lo < full[0]) { hi += full[0] - lo; lo = full[0]; }
  if (hi > full[1]) { lo -= hi - full[1]; hi = full[1]; }
  setView(name, lo, hi, full);
}

function panBy(name, frac, full) {
  const v = view(name, full);
  const d = (v[1] - v[0]) * frac;
  let lo = v[0] + d, hi = v[1] + d;
  if (lo < full[0]) { hi += full[0] - lo; lo = full[0]; }
  if (hi > full[1]) { lo -= hi - full[1]; hi = full[1]; }
  setView(name, lo, hi, full);
}

function yWin(name, auto) {
  const v = S.viewY[name];
  return v ? v.slice() : auto;
}

function atAxisY(name, f) {
  const a = S.axes[name];
  if (!a) { return 0; }
  return a.yl[1] - (a.yl[1] - a.yl[0]) * f;
}

function zoomYAt(name, factor, at) {
  const a = S.axes[name];
  if (!a || a.flat) { return; }
  const h = (a.yl[1] - a.yl[0]) * factor;
  const pivot = atAxisY(name, at);
  S.viewY[name] = [pivot - h * (1 - at), pivot + h * at];
}

function panYBy(name, frac) {
  const a = S.axes[name];
  if (!a || a.flat) { return; }
  const d = (a.yl[1] - a.yl[0]) * frac;
  S.viewY[name] = [a.yl[0] + d, a.yl[1] + d];
}

function spanOf(list) {
  let lo = Infinity, hi = -Infinity;
  list.forEach((ys) => ys.forEach((v) => {
    if (v < lo) { lo = v; }
    if (v > hi) { hi = v; }
  }));
  if (!isFinite(lo)) { return [0, 1]; }
  if (hi <= lo) { hi = lo + Math.max(1e-6, Math.abs(lo) * 1e-5); }
  const p = (hi - lo) * 0.07;
  return [lo - p, hi + p];
}

function logSpan(list, floor) {
  let lo = Infinity, hi = -Infinity;
  list.forEach((ys) => ys.forEach((v) => {
    const u = Math.log10(Math.max(v, floor));
    if (u < lo) { lo = u; }
    if (u > hi) { hi = u; }
  }));
  if (!isFinite(lo)) { return [-1, 1]; }
  return [Math.floor(lo), Math.ceil(hi)];
}

function cut(xs, lim) {
  let a = 0, b = xs.length - 1;
  while (a < xs.length && xs[a] < lim[0]) { a++; }
  while (b > 0 && xs[b] > lim[1]) { b--; }
  return [a > 0 ? a - 1 : 0, b < xs.length - 1 ? b + 1 : xs.length - 1];
}

function line(xs, ys, box, xl, yl, opt) {
  const o = opt || {};
  const a = o.from === undefined ? 0 : o.from;
  const b = o.to === undefined ? xs.length - 1 : o.to;
  const dx = xl[1] - xl[0], dy = yl[1] - yl[0], off = o.yoff || 0;
  const h = o.h === undefined ? box.h : o.h;
  let d = '';
  for (let i = a; i <= b; i++) {
    const v = o.log ? Math.log10(Math.max(ys[i], o.floor)) : ys[i];
    d += (i === a ? 'M' : 'L') + (box.w * (xs[i] - xl[0]) / dx).toFixed(1) + ' '
      + (off + h - h * (v - yl[0]) / dy).toFixed(1);
  }
  return d;
}

function bandPath(xs, los, his, box, xl, yl, a, b) {
  const dx = xl[1] - xl[0], dy = yl[1] - yl[0];
  let d = '', i;
  for (i = a; i <= b; i++) {
    d += (i === a ? 'M' : 'L') + (box.w * (xs[i] - xl[0]) / dx).toFixed(1) + ' '
      + (box.h - box.h * (his[i] - yl[0]) / dy).toFixed(1);
  }
  for (i = b; i >= a; i--) {
    d += 'L' + (box.w * (xs[i] - xl[0]) / dx).toFixed(1) + ' '
      + (box.h - box.h * (los[i] - yl[0]) / dy).toFixed(1);
  }
  return d + 'Z';
}

function markPath(ts, ys, box, xl, yl) {
  const dx = xl[1] - xl[0], dy = yl[1] - yl[0];
  let d = '';
  for (let i = 0; i < ts.length; i++) {
    if (ts[i] < xl[0] || ts[i] > xl[1]) { continue; }
    const px = (box.w * (ts[i] - xl[0]) / dx).toFixed(1);
    const py = (box.h - box.h * (ys[i] - yl[0]) / dy).toFixed(1);
    d += 'M' + px + ' ' + py + 'L' + px + ' ' + py;
  }
  return d;
}

function ticks(host, lim, n, f) {
  if (!host) { return; }
  host.innerHTML = '';
  for (let i = 0; i <= n; i++) {
    const d = document.createElement('div');
    d.textContent = f(lim[0] + (lim[1] - lim[0]) * i / n);
    host.appendChild(d);
  }
}

function labels(host, list) {
  host.innerHTML = '';
  list.forEach((v) => {
    const d = document.createElement('div');
    d.textContent = v;
    host.appendChild(d);
  });
}

function legend(name, pads) {
  const host = $('[data-legend="' + name + '"]');
  if (!host) { return; }
  host.innerHTML = '';
  pads.forEach((p) => {
    const d = document.createElement('div');
    d.innerHTML = chip(p) + 'PAD ' + p;
    host.appendChild(d);
  });
}

function axes(name, box, xl, yl, opt) {
  const o = opt || {};
  S.axes[name] = { box, xl, yl, log: !!o.log, fx: o.fx, fy: o.fy,
                   date: !!o.date, flat: !!o.flat };
}

function atX(name, frac) {
  const a = S.axes[name];
  return a ? a.xl[0] + (a.xl[1] - a.xl[0]) * frac : 0;
}

function atY(name, frac) {
  const a = S.axes[name];
  if (!a) { return 0; }
  const v = a.yl[1] - (a.yl[1] - a.yl[0]) * frac;
  return a.log ? Math.pow(10, v) : v;
}

function sayX(name, v) {
  const a = S.axes[name];
  if (!a) { return ''; }
  if (a.date) {
    return new Date(v * 1000).toLocaleString('en-GB',
      { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' });
  }
  return a.fx ? a.fx(v) : v.toFixed(2);
}

function sayY(name, v) {
  const a = S.axes[name];
  if (!a || !a.fy) { return ''; }
  return a.fy(v);
}

function overlayOf(svg) {
  const out = {};
  ['marks', 'cursor'].forEach((kind) => {
    let g = $('[data-over="' + kind + '"]', svg);
    if (!g) {
      g = el('g', { 'data-over': kind, 'pointer-events': 'none' });
    }
    svg.appendChild(g);
    g.innerHTML = '';
    out[kind] = g;
  });
  return out;
}

function drawOverlay(name) {
  const svg = $('[data-zoom="' + name + '"]');
  const a = S.axes[name];
  if (!svg || !a) { return; }
  const layer = overlayOf(svg);
  let g = layer.marks;
  const box = a.box;
  const put = (x, y, text, colour, above) => {
    const w = Math.max(46, text.length * 5.4 + 8);
    const left = x + w + 6 > box.w ? x - w - 6 : x + 6;
    const top = above ? Math.max(2, y - 20) : Math.min(box.h - 16, y + 6);
    g.appendChild(el('rect', { x: left.toFixed(1), y: top.toFixed(1),
      width: w.toFixed(1), height: '14', rx: '2', fill: '#ffffff',
      stroke: colour, 'stroke-width': '0.8', opacity: '0.94' }));
    const t = el('text', { x: (left + 4).toFixed(1), y: (top + 10.2).toFixed(1),
      fill: colour, 'font-size': '9.5',
      'font-family': 'IBM Plex Mono, ui-monospace, monospace' });
    t.textContent = text;
    g.appendChild(t);
  };

  (S.marks[name] || []).forEach((m) => {
    const x = box.w * (m.x - a.xl[0]) / (a.xl[1] - a.xl[0]);
    if (x < -2 || x > box.w + 2) { return; }
    g.appendChild(el('path', { d: 'M' + x.toFixed(1) + ' 0L' + x.toFixed(1)
      + ' ' + box.h, stroke: '#a4553a', 'stroke-width': '0.9',
      'stroke-dasharray': '3 3', fill: 'none' }));
    if (m.y !== null) {
      const yv = a.log ? Math.log10(Math.max(m.y, 1e-12)) : m.y;
      const y = box.h - box.h * (yv - a.yl[0]) / (a.yl[1] - a.yl[0]);
      g.appendChild(el('path', { d: 'M' + (x - 5).toFixed(1) + ' ' + y.toFixed(1)
        + 'L' + (x + 5).toFixed(1) + ' ' + y.toFixed(1) + 'M' + x.toFixed(1)
        + ' ' + (y - 5).toFixed(1) + 'L' + x.toFixed(1) + ' ' + (y + 5).toFixed(1),
        stroke: '#a4553a', 'stroke-width': '1.4', fill: 'none' }));
      put(x, y, m.label, '#a4553a', true);
    }
  });

  const c = S.cursor[name];
  if (!c || S.drag) { return; }
  g = layer.cursor;
  const x = c.fx * box.w, y = c.fy * box.h;
  g.appendChild(el('path', { d: 'M' + x.toFixed(1) + ' 0L' + x.toFixed(1) + ' '
    + box.h + 'M0 ' + y.toFixed(1) + 'L' + box.w + ' ' + y.toFixed(1),
    stroke: '#6f6a5f', 'stroke-width': '0.7', 'stroke-dasharray': '2 3',
    fill: 'none', opacity: '0.75' }));
  const label = sayX(name, atX(name, c.fx))
    + (a.fy ? '   ' + sayY(name, atY(name, c.fy)) : '');
  put(x, y, label, '#1d1b17', false);
}

/* -------------------------------------------------------------- interaction */

function fullOf(name) {
  if (name === 'raw') {
    return S.record ? [0, S.record.seconds] : [0, 1];
  }
  if (name === 'det') {
    return S.record ? S.record.window : [0, 1];
  }
  if (name === 'psd' || name === 'fft') {
    const R = S.record;
    if (!R) { return [0, 60]; }
    const f = R.spectra[name].f;
    return [f[0], f[f.length - 1]];
  }
  if (name === 'emd') { return S.record ? S.record.emd.span : [0, 1]; }
  if (name === 'camp') { return [0, 1]; }
  return [0, 1];
}

function afterView(name) {
  if (name === 'raw' || name === 'det') { scheduleFetch(name); }
  draw();
}

let fetchTimer = {};
function scheduleFetch(name) {
  clearTimeout(fetchTimer[name]);
  fetchTimer[name] = setTimeout(() => {
    const win = view(name, fullOf(name));
    S.pads.forEach((p) => (name === 'raw' ? loadRaw : loadTrace)(p, win));
  }, 180);
}

function wire(svg, name) {
  const frac = (ev) => {
    const r = svg.getBoundingClientRect();
    return r.width ? clip((ev.clientX - r.left) / r.width, 0, 1) : 0;
  };
  const fracY = (ev) => {
    const r = svg.getBoundingClientRect();
    return r.height ? clip((ev.clientY - r.top) / r.height, 0, 1) : 0;
  };
  svg.addEventListener('mousedown', (ev) => {
    ev.preventDefault();
    const panning = (S.mode === 'pan') !== (ev.shiftKey || ev.button === 1);
    const a = S.axes[name];
    S.drag = { name, a: frac(ev), b: frac(ev),
               ay: fracY(ev), by: fracY(ev), pan: panning,
               view: view(name, fullOf(name)),
               viewY: a ? a.yl.slice() : null };
    svg.classList.toggle('panning', S.drag.pan);
  });
  svg.addEventListener('mousemove', (ev) => {
    const d = S.drag;
    if (d && d.name === name) {
      const f = frac(ev), fy = fracY(ev);
      if (d.pan) {
        const full = fullOf(name);
        const w = d.view[1] - d.view[0];
        const shift = (d.a - f) * w;
        let lo = d.view[0] + shift, hi = d.view[1] + shift;
        if (lo < full[0]) { hi += full[0] - lo; lo = full[0]; }
        if (hi > full[1]) { lo -= hi - full[1]; hi = full[1]; }
        setView(name, lo, hi, full);
        const a = S.axes[name];
        if (d.viewY && a && !a.flat) {
          const dy = (d.viewY[1] - d.viewY[0]) * (fy - d.ay);
          S.viewY[name] = [d.viewY[0] + dy, d.viewY[1] + dy];
        }
        draw();
      } else {
        d.b = f;
        d.by = fy;
        draw();
      }
    } else {
      const r = svg.getBoundingClientRect();
      S.cursor[name] = { fx: frac(ev),
        fy: r.height ? clip((ev.clientY - r.top) / r.height, 0, 1) : 0 };
      if (name === 'camp') { hoverCampaign(frac(ev)); } else { draw(); }
    }
  });
  const end = () => {
    const d = S.drag;
    svg.classList.remove('panning');
    if (!d || d.name !== name) { return; }
    S.drag = null;
    if (d.pan) { afterView(name); return; }
    const lo = Math.min(d.a, d.b), hi = Math.max(d.a, d.b);
    const ylo = Math.min(d.ay, d.by), yhi = Math.max(d.ay, d.by);
    const wideEnough = hi - lo >= 0.012;
    const tallEnough = yhi - ylo >= 0.05 && d.viewY;
    if (!wideEnough && !tallEnough) { draw(); return; }
    if (wideEnough) {
      const full = fullOf(name);
      const v = d.view;
      setView(name, v[0] + (v[1] - v[0]) * lo, v[0] + (v[1] - v[0]) * hi, full);
    }
    const a = S.axes[name];
    if (tallEnough && a && !a.flat) {
      const t = d.viewY[1], b2 = d.viewY[0];
      S.viewY[name] = [t - (t - b2) * yhi, t - (t - b2) * ylo];
    }
    afterView(name);
  };
  svg.addEventListener('mouseup', end);
  svg.addEventListener('mouseleave', () => {
    if (S.drag && S.drag.name === name) { end(); }
    delete S.cursor[name];
    if (name === 'camp' && S.hover >= 0) { S.hover = -1; }
    draw();
  });
  svg.addEventListener('click', (ev) => {
    if (!S.markMode) { return; }
    ev.stopPropagation();
    const a = S.axes[name];
    if (!a) { return; }
    const r = svg.getBoundingClientRect();
    const fx = frac(ev);
    const fy = r.height ? clip((ev.clientY - r.top) / r.height, 0, 1) : 0;
    const x = atX(name, fx), y = atY(name, fy);
    const list = S.marks[name] || (S.marks[name] = []);
    const near = list.findIndex((m) =>
      Math.abs((m.x - a.xl[0]) / (a.xl[1] - a.xl[0]) - fx) < 0.012);
    if (near >= 0) { list.splice(near, 1); } else {
      list.push({ x, y, label: sayX(name, x) + '   ' + sayY(name, y) });
    }
    draw();
  });
  svg.addEventListener('dblclick', () => {
    delete S.view[name];
    delete S.viewY[name];
    afterView(name);
  });
  svg.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    const k = ev.deltaY > 0 ? 1.35 : 1 / 1.35;
    if (!ev.shiftKey) { zoomAt(name, k, frac(ev), fullOf(name)); }
    if (!ev.altKey) { zoomYAt(name, k, fracY(ev)); }
    afterView(name);
  }, { passive: false });
  svg.addEventListener('contextmenu', (ev) => {
    ev.preventDefault();
    zoomAt(name, 1.6, frac(ev), fullOf(name));
    zoomYAt(name, 1.6, fracY(ev));
    afterView(name);
  });
}

const ICON = {
  camera: '<rect x="2.5" y="4.8" width="13" height="9.2" rx="1.4"/>'
        + '<path d="M6.5 4.8l1-2h3l1 2"/><circle cx="9" cy="9.4" r="2.5"/>',
  zoom: '<circle cx="7.8" cy="7.8" r="4.6"/><path d="M11.2 11.2L15.5 15.5"/>'
      + '<path d="M5.6 5.6h4.4v4.4"/>',
  pan: '<path d="M9 2.5v13M2.5 9h13"/>'
     + '<path d="M9 2.5l-2 2M9 2.5l2 2M9 15.5l-2-2M9 15.5l2-2"/>'
     + '<path d="M2.5 9l2-2M2.5 9l2 2M15.5 9l-2-2M15.5 9l-2 2"/>',
  plus: '<circle cx="7.6" cy="7.6" r="4.6"/><path d="M11 11L15.5 15.5"/>'
      + '<path d="M5.6 7.6h4M7.6 5.6v4"/>',
  minus: '<circle cx="7.6" cy="7.6" r="4.6"/><path d="M11 11L15.5 15.5"/>'
       + '<path d="M5.6 7.6h4"/>',
  home: '<path d="M2.8 8.2L9 2.8l6.2 5.4"/><path d="M4.6 8v7h8.8V8"/>',
  grow: '<path d="M2.6 6.6v-4h4M15.4 6.6v-4h-4M2.6 11.4v4h4M15.4 11.4v4h-4"/>',
  mark: '<path d="M9 2.2v13.6M2.2 9h13.6"/><circle cx="9" cy="9" r="3.2"/>',
  clear: '<path d="M4 4l10 10M14 4L4 14"/>',
  base: '<path d="M2.5 12.6h13"/><path d="M2.5 8q3-5 6 0t6 0"/>'
};

function icon(name) {
  return '<svg width="17" height="17" viewBox="0 0 18 18" fill="none" '
    + 'stroke="currentColor" stroke-width="1.25" stroke-linecap="round" '
    + 'stroke-linejoin="round">' + ICON[name] + '</svg>';
}

function setMode(m) {
  S.mode = m;
  $$('.ctl button[data-mode]').forEach(function (b) {
    const k = b.dataset.mode;
    b.classList.toggle('on', k === 'mark' ? S.markMode : k === m);
  });
  $$('.sv').forEach(function (x) {
    x.classList.toggle('grab', m === 'pan');
    x.classList.toggle('marking', S.markMode);
  });
}

const SANS = "'IBM Plex Sans', Segoe UI, system-ui, sans-serif";
const MONO = "'IBM Plex Mono', Consolas, ui-monospace, monospace";
const SERIF = "'Source Serif 4', Georgia, serif";

function words(g, x, y, text, opt) {
  const o = opt || {};
  const t = el('text', {
    x: String(x), y: String(y), fill: o.fill || '#1d1b17',
    'font-size': String(o.size || 10),
    'font-family': o.font || SANS,
    'text-anchor': o.anchor || 'start'
  });
  if (o.weight) { t.setAttribute('font-weight', String(o.weight)); }
  if (o.rotate) { t.setAttribute('transform', 'rotate(' + o.rotate + ' ' + x + ' ' + y + ')'); }
  t.textContent = text;
  g.appendChild(t);
  return t;
}

function exportSVG(name) {
  const card = $('[data-panel="' + name + '"]');
  const svg = $('[data-zoom="' + name + '"]');
  if (!card || !svg) { return null; }
  const r = svg.getBoundingClientRect();
  const pw = Math.round(r.width), ph = Math.round(r.height);
  const yLab = $('.yl', card), yc = $('[data-yc="' + name + '"]', card);
  const xr = $('[data-xr="' + name + '"]', card);
  const ax = $('.ax', card);
  const legendHost = $('[data-legend="' + name + '"]', card);
  const keyHost = name === 'det' ? $('#det-key') : null;

  const yTicks = yc ? $$('div', yc).map((d) => d.textContent) : [];
  const xTicks = xr ? $$('div', xr).map((d) => d.textContent) : [];
  const chips = [];
  const readChip = (d) => {
    const i = $('i', d), u = $('u', d);
    if (u) {
      return { colour: u.style.borderTopColor || u.style.borderColor || '#1d1b17',
               line: true, dashed: u.style.borderTopStyle === 'dashed',
               text: d.textContent.trim() };
    }
    return { colour: i ? i.style.background : '#1d1b17',
             line: false, dashed: false, text: d.textContent.trim() };
  };
  if (legendHost) { $$('div', legendHost).forEach((d) => chips.push(readChip(d))); }
  if (keyHost) { $$('div', keyHost).forEach((d) => chips.push(readChip(d))); }

  const left = 78, right = 18, top = 44;
  const xtH = xTicks.length ? 15 : 0;
  const axH = ax && ax.textContent.trim() ? 15 : 0;
  const legH = chips.length ? 18 : 0;
  const W = left + pw + right;
  const H = top + ph + xtH + axH + legH + 12;

  const out = el('svg', { xmlns: 'http://www.w3.org/2000/svg',
    width: String(W), height: String(H), viewBox: '0 0 ' + W + ' ' + H });
  out.appendChild(el('rect', { x: 0, y: 0, width: W, height: H, fill: '#ffffff' }));

  const title = $('.pt', card);
  words(out, left, 20, title ? title.textContent : name,
        { size: 13, font: SERIF, weight: 600 });
  const note = $('.mono.small', card);
  if (note && note.textContent.trim()) {
    words(out, W - right, 20, note.textContent.trim(),
          { size: 9.5, font: MONO, fill: '#6f6a5f', anchor: 'end' });
  }
  const when = S.record ? S.record.record.file + '   PAD '
    + S.pads.join(', ') + '   ' + S.engine : '';
  if (when) {
    words(out, left, 34, when, { size: 9, font: MONO, fill: '#989285' });
  }

  if (yLab && yLab.textContent.trim()) {
    words(out, 13, top + ph / 2, yLab.textContent.trim(),
          { size: 9.5, fill: '#6f6a5f', anchor: 'middle', rotate: -90 });
  }
  const spread = yc && yc.classList.contains('spread');
  yTicks.forEach((tk, i) => {
    const f = yTicks.length > 1 ? i / (yTicks.length - 1) : 0.5;
    const y = spread
      ? top + ph * (yTicks.length - i - 0.5) / yTicks.length
      : top + ph - ph * f;
    words(out, left - 7, y + 3.2, tk,
          { size: 9, font: MONO, fill: '#989285', anchor: 'end' });
  });

  const plot = el('g', { transform: 'translate(' + left + ',' + top + ')' });
  ['[data-el="' + name + '-body"]', '[data-over="marks"]'].forEach((sel) => {
    const src = $(sel, svg);
    if (src) { plot.appendChild(src.cloneNode(true)); }
  });
  out.appendChild(plot);
  out.appendChild(el('rect', { x: left, y: top, width: pw, height: ph,
    fill: 'none', stroke: '#ded8cc', 'stroke-width': '1' }));

  xTicks.forEach((tk, i) => {
    const f = xTicks.length > 1 ? i / (xTicks.length - 1) : 0.5;
    const anchor = i === 0 ? 'start' : i === xTicks.length - 1 ? 'end' : 'middle';
    words(out, left + pw * f, top + ph + 11, tk,
          { size: 9, font: MONO, fill: '#989285', anchor });
  });
  if (axH) {
    words(out, left + pw / 2, top + ph + xtH + 11, ax.textContent.trim(),
          { size: 9.5, fill: '#6f6a5f', anchor: 'middle' });
  }
  let cx = left;
  chips.forEach((c) => {
    const y = top + ph + xtH + axH + 10;
    if (c.line) {
      out.appendChild(el('path', { d: 'M' + cx + ' ' + (y - 2) + 'h13',
        stroke: c.colour, 'stroke-width': '2', fill: 'none',
        'stroke-dasharray': c.dashed ? '4 2' : 'none' }));
    } else {
      out.appendChild(el('rect', { x: String(cx), y: String(y - 6),
        width: '8', height: '8', rx: '2', fill: c.colour }));
    }
    const t = words(out, cx + (c.line ? 17 : 12), y + 1, c.text,
                    { size: 9, font: MONO, fill: '#6f6a5f' });
    cx += (c.line ? 17 : 12) + c.text.length * 5.2 + 14;
    if (cx > W - right) { t.remove(); }
  });
  return { node: out, w: W, h: H };
}

function savePNG(name) {
  const built = exportSVG(name);
  if (!built) { return; }
  const r = { width: built.w, height: built.h };
  const src = 'data:image/svg+xml;charset=utf-8,'
    + encodeURIComponent(new XMLSerializer().serializeToString(built.node));
  const img = new Image();
  img.onload = function () {
    const c = document.createElement('canvas');
    c.width = Math.round(r.width * 2);
    c.height = Math.round(r.height * 2);
    const g = c.getContext('2d');
    g.fillStyle = '#ffffff';
    g.fillRect(0, 0, c.width, c.height);
    g.drawImage(img, 0, 0, c.width, c.height);
    c.toBlob(function (blob) {
      const a = document.createElement('a');
      const rec = S.record ? S.record.record.file.replace(/[^A-Za-z0-9]+/g, '-')
                           : 'lightrack';
      a.download = rec + '-' + name + '.png';
      a.href = URL.createObjectURL(blob);
      a.click();
      setTimeout(function () { URL.revokeObjectURL(a.href); }, 4000);
    });
  };
  img.onerror = function () { toast('could not make a picture of that chart'); };
  img.src = src;
}

function controls(name) {
  const host = $('[data-ctl="' + name + '"]');
  if (!host) { return; }
  host.innerHTML = '';
  const mk = (glyph, title, fn, mode) => {
    const b = document.createElement('button');
    b.innerHTML = icon(glyph);
    b.title = title;
    if (mode) {
      b.dataset.mode = mode;
      if (S.mode === mode) { b.classList.add('on'); }
    }
    b.onclick = fn;
    host.appendChild(b);
  };
  mk('camera', 'save this chart as a picture', () => savePNG(name));
  mk('zoom', 'drag to zoom into a range', () => setMode('zoom'), 'zoom');
  mk('pan', 'drag to move about', () => setMode('pan'), 'pan');
  mk('plus', 'zoom in', () => {
    zoomAt(name, 1 / 1.6, 0.5, fullOf(name));
    zoomYAt(name, 1 / 1.6, 0.5);
    afterView(name);
  });
  mk('minus', 'zoom out', () => {
    zoomAt(name, 1.6, 0.5, fullOf(name));
    zoomYAt(name, 1.6, 0.5);
    afterView(name);
  });
  mk('home', 'show everything', () => {
    delete S.view[name];
    delete S.viewY[name];
    afterView(name);
  });
  mk('mark', 'click the chart to mark a place, click a mark to drop it',
     () => { S.markMode = !S.markMode; setMode(S.mode); }, 'mark');
  mk('clear', 'take the marks off this chart',
     () => { S.marks[name] = []; draw(); });
  if (name === 'raw') {
    mk('base', 'keep or remove the baseline', () => {
      S.keepBase = !S.keepBase;
      S.pads.forEach((p) => loadRaw(p, view('raw', fullOf('raw'))));
    });
  }
  mk('grow', 'fill the window', () => setBig(name));
}

/* ------------------------------------------------------------------ drawing */

function drawRaw() {
  const svg = $('[data-zoom="raw"]');
  const box = fitBox(svg);
  const host = $('[data-el="raw-body"]');
  host.innerHTML = '';
  const sets = S.pads.map((p) => S.raw[p]).filter(Boolean);
  if (!sets.length) { return; }
  const lim = view('raw', fullOf('raw'));
  const ylim = yWin('raw',
    spanOf(sets.map((d) => d.hi).concat(sets.map((d) => d.lo))));
  const T = S.record && S.record.trace;
  if (T && T.first_s !== undefined) {
    const a = clip((T.first_s - lim[0]) / (lim[1] - lim[0]), 0, 1);
    const b = clip((T.last_s - lim[0]) / (lim[1] - lim[0]), 0, 1);
    if (b > a) {
      host.appendChild(el('rect', { x: (a * box.w).toFixed(1), y: 0,
        width: ((b - a) * box.w).toFixed(1), height: box.h,
        fill: COLOUR[S.primary], opacity: '0.06' }));
    }
  }
  S.pads.forEach((p) => {
    const d = S.raw[p];
    if (!d) { return; }
    if (d.mode === 'points') {
      host.appendChild(el('path', { d: line(d.t, d.hi, box, lim, ylim, {}),
        fill: 'none', stroke: COLOUR[p], 'stroke-width': '1.2',
        'stroke-dasharray': DASH[p],
        'vector-effect': 'non-scaling-stroke' }));
    } else {
      host.appendChild(el('path', {
        d: bandPath(d.t, d.lo, d.hi, box, lim, ylim, 0, d.t.length - 1),
        fill: COLOUR[p], opacity: S.pads.length > 1 ? '0.5' : '0.88' }));
      if (S.pads.length > 1) {
        host.appendChild(el('path', { d: line(d.t, d.hi, box, lim, ylim, {}),
          fill: 'none', stroke: COLOUR[p], 'stroke-width': '0.9',
          'stroke-dasharray': DASH[p], opacity: '0.95',
          'vector-effect': 'non-scaling-stroke' }));
      }
    }
  });
  const unit = sets[0].unit || 'nm';
  axes('raw', box, lim, ylim, {
    fx: (v) => v.toFixed(3) + ' s',
    fy: (v) => v.toFixed(unit === 'nm' ? 4 : 1) + ' ' + unit });
  ticks($('[data-yc="raw"]'), ylim, 3, (v) => v.toFixed(unit === 'nm' ? 4 : 0));
  ticks($('[data-xr="raw"]'), lim, 6, (v) => v.toFixed(lim[1] - lim[0] < 6 ? 2 : 0));
  legend('raw', S.pads);
  const d0 = S.raw[S.primary];
  $('[data-panel="raw"] .yl').textContent =
    unit === 'nm' ? 'raw [nm]' : 'baseline removed [pm]';
  $('[data-panel="raw"] .pt').textContent =
    unit === 'nm' ? 'Raw signal, baseline kept' : 'Raw signal, baseline removed';
  $('#raw-note').textContent = d0
    ? (d0.mode === 'points' ? 'every sample' : 'min and max per bin')
      + (unit === 'nm' ? ''
         : ', baseline ' + d0.base.toFixed(4) + ' nm on PAD ' + S.primary)
    : '';
}

function drawDetection() {
  const svg = $('[data-zoom="det"]');
  const box = fitBox(svg);
  const host = $('[data-el="det-body"]');
  host.innerHTML = '';
  $('#det-key').innerHTML = '';
  const sets = S.pads.map((p) => [p, S.det[p]]).filter((x) => x[1] && x[1].t);
  if (!sets.length) { return; }
  const lim = view('det', fullOf('det'));
  const ylim = yWin('det', spanOf(sets.map((x) => x[1].y)));
  const only = sets.length === 1;
  sets.forEach(([p, T]) => {
    if (only) {
      (T.guides || []).forEach((g) => host.appendChild(el('path', {
        d: line(T.t, g.v, box, lim, ylim, {}), fill: 'none', stroke: g.colour,
        'stroke-width': '1', opacity: '0.7', 'vector-effect': 'non-scaling-stroke' })));
    }
    host.appendChild(el('path', { d: line(T.t, T.y, box, lim, ylim, {}),
      fill: 'none', stroke: COLOUR[p], 'stroke-width': '1',
      'stroke-dasharray': only ? 'none' : DASH[p],
      opacity: only ? '1' : '0.75', 'vector-effect': 'non-scaling-stroke' }));
    (T.marks || []).forEach((m, k) => {
      if (!m.t.length) { return; }
      host.appendChild(el('path', {
        d: markPath(m.t, m.y, box, lim, ylim), fill: 'none',
        stroke: only ? m.colour : COLOUR[p], 'stroke-width': k ? '8' : '6',
        'stroke-linecap': 'round', opacity: k ? '0.9' : '1',
        'vector-effect': 'non-scaling-stroke' }));
    });
  });
  const key = $('#det-key');
  const T0 = sets[0][1];
  $('#det-units').textContent = T0.units || 'signal';
  if (only) {
    (T0.marks || []).forEach((m) => {
      if (!m.t.length) { return; }
      const d = document.createElement('div');
      d.innerHTML = '<i style="background:' + m.colour + '"></i>' + m.name + ' (' + m.t.length + ')';
      key.appendChild(d);
    });
    (T0.guides || []).forEach((g) => {
      const d = document.createElement('div');
      d.innerHTML = '<u style="border-color:' + g.colour + '"></u>' + g.name;
      key.appendChild(d);
    });
  } else {
    sets.forEach(([p, T]) => {
      const n = (T.marks || []).reduce((a, m) => a + m.t.length, 0);
      const d = document.createElement('div');
      d.innerHTML = chip(p) + 'PAD ' + p + ' (' + n + ')';
      key.appendChild(d);
    });
  }
  axes('det', box, lim, ylim, {
    fx: (v) => v.toFixed(3) + ' s', fy: (v) => v.toFixed(1) + ' pm' });
  ticks($('[data-yc="det"]'), ylim, 3, (v) => v.toFixed(0));
  ticks($('[data-xr="det"]'), lim, 6, (v) => v.toFixed(lim[1] - lim[0] < 6 ? 2 : 1));
}

function drawSpectrum(name, floor) {
  const svg = $('[data-zoom="' + name + '"]');
  const box = fitBox(svg);
  const host = $('[data-el="' + name + '-body"]');
  host.innerHTML = '';
  const R = S.record;
  if (!R) { return; }
  const sets = S.pads.map((p) => [p, (S.record.pads || {})[p]]).filter((x) => x[1]);
  const use = sets.length ? sets : [[S.primary, R.spectra]];
  const lim = view(name, fullOf(name));
  const f = R.spectra[name].f;
  const c = cut(f, lim);
  const ylim = yWin(name, logSpan(use.map((x) => x[1][name].v), floor));
  use.forEach(([p, sp]) => {
    host.appendChild(el('path', {
      d: line(sp[name].f, sp[name].v, box, lim, ylim,
        { log: true, floor, from: c[0], to: c[1] }),
      fill: 'none', stroke: COLOUR[p], 'stroke-width': '1.2',
      'stroke-dasharray': use.length > 1 ? DASH[p] : 'none',
      opacity: use.length > 1 ? '0.85' : '1',
      'vector-effect': 'non-scaling-stroke' }));
  });
  axes(name, box, lim, ylim, { log: true,
    fx: (v) => v.toFixed(2) + ' Hz', fy: (v) => v.toPrecision(3) });
  ticks($('[data-yc="' + name + '"]'), ylim, 4, (x) => '1e' + x.toFixed(0));
  ticks($('[data-xr="' + name + '"]'), lim, 4,
    (x) => x.toFixed(lim[1] - lim[0] < 8 ? 1 : 0));
}

function drawModes() {
  const svg = $('[data-zoom="emd"]');
  const box = fitBox(svg);
  const host = $('[data-el="emd-body"]');
  host.innerHTML = '';
  const R = S.record;
  if (!R) { return; }
  const sets = S.pads.map((p) => [p, ((R.pads || {})[p] || {}).emd || (p === R.pad ? R.emd : null)])
    .filter((x) => x[1] && x[1].imf && x[1].imf.length);
  if (!sets.length) { return; }
  const lim = view('emd', fullOf('emd'));
  const lanes = sets[0][1].imf.length;
  const lane = box.h / lanes;
  for (let k = 1; k < lanes; k++) {
    host.appendChild(el('path', { d: 'M0 ' + (k * lane).toFixed(1) + 'L' + box.w
      + ' ' + (k * lane).toFixed(1), stroke: '#ded8cc', 'stroke-width': '1',
      fill: 'none', 'vector-effect': 'non-scaling-stroke' }));
  }
  for (let k = 0; k < lanes; k++) {
    let span = 1e-9;
    sets.forEach(([, E]) => {
      const m = E.imf[k] || [];
      const c = cut(E.t, lim);
      for (let i = c[0]; i <= c[1]; i++) {
        if (Math.abs(m[i]) > span) { span = Math.abs(m[i]); }
      }
    });
    sets.forEach(([p, E]) => {
      const m = E.imf[k];
      if (!m) { return; }
      const c = cut(E.t, lim);
      host.appendChild(el('path', {
        d: line(E.t, m, box, lim, [-span, span],
          { yoff: k * lane + 2.5, h: lane - 5, from: c[0], to: c[1] }),
        fill: 'none', stroke: COLOUR[p], 'stroke-width': '1',
        'stroke-dasharray': sets.length > 1 ? DASH[p] : 'none',
        opacity: sets.length > 1 ? '0.8' : '1',
        'vector-effect': 'non-scaling-stroke' }));
    });
  }
  axes('emd', box, lim, [0, 1], { fx: (v) => v.toFixed(3) + ' s', flat: true });
  labels($('[data-yc="emdlab"]'), sets[0][1].imf.map((_, k) => 'IMF ' + (k + 1)));
  ticks($('[data-xr="emd"]'), lim, 4, (v) => v.toFixed(1));
  $('#emd-note').textContent = sets[0][1].span[0].toFixed(0) + ' to '
    + sets[0][1].span[1].toFixed(0) + ' s';
}

function campLayout() {
  const D = S.series[S.primary];
  if (!D || !D.series.length) { return null; }
  const P = D.panels.panels, B = D.panels.breaks;
  const total = P[P.length - 1].x1 - P[0].x0;
  const toX = (epoch) => {
    for (let i = 0; i < P.length; i++) {
      if (epoch <= P[i].t1 || i === P.length - 1) {
        const f = clip((epoch - P[i].t0) / (P[i].t1 - P[i].t0), 0, 1);
        return (P[i].x0 + (P[i].x1 - P[i].x0) * f) / total;
      }
    }
    return 0;
  };
  const toEpoch = (x) => {
    const u = x * total;
    for (let i = 0; i < P.length; i++) {
      if (u <= P[i].x1 || i === P.length - 1) {
        const f = clip((u - P[i].x0) / (P[i].x1 - P[i].x0), 0, 1);
        return P[i].t0 + (P[i].t1 - P[i].t0) * f;
      }
    }
    return P[0].t0;
  };
  return { rows: D.series, P, B, total, toX, toEpoch };
}

function hoverCampaign(frac) {
  const L = campLayout();
  if (!L) { return; }
  const v = view('camp', [0, 1]);
  const x = v[0] + (v[1] - v[0]) * frac;
  const t = L.toEpoch(x);
  const rows = L.rows;
  let a = 0, b = rows.length - 1;
  while (a < b) {
    const m = (a + b) >> 1;
    if (rows[m].epoch < t) { a = m + 1; } else { b = m; }
  }
  if (a > 0 && Math.abs(rows[a - 1].epoch - t) < Math.abs(rows[a].epoch - t)) { a--; }
  if (a !== S.hover) { S.hover = a; draw(); }
}

function drawCampaign() {
  const svg = $('[data-zoom="camp"]');
  const box = fitBox(svg);
  const host = $('[data-el="camp-body"]');
  host.innerHTML = '';
  const L = campLayout();
  if (!L) { $('#camp-foot').textContent = ''; return; }
  const v = view('camp', [0, 1]);
  const px = (x) => box.w * (x - v[0]) / (v[1] - v[0]);

  const pads = S.pads.filter((p) => S.series[p]);
  let top = 1e-9, bot = Infinity;
  pads.forEach((p) => S.series[p].series.forEach((r) => {
    if (r.mean + r.sd > top) { top = r.mean + r.sd; }
    if (r.mean < bot) { bot = r.mean; }
  }));
  const lim = yWin('camp', [0, top * 1.06]);
  const cy = (val) => box.h - box.h * (val - lim[0]) / (lim[1] - lim[0]);

  L.B.forEach((b) => {
    const a = px(b.x0 / L.total), c = px(b.x1 / L.total);
    if (c < 0 || a > box.w) { return; }
    const mid = (a + c) / 2;
    host.appendChild(el('rect', { x: a.toFixed(1), y: 0,
      width: Math.max(1, c - a).toFixed(1), height: box.h, fill: '#f7f5f0' }));
    [0, box.h].forEach((yy) => {
      [-3, 3].forEach((off) => {
        host.appendChild(el('path', {
          d: 'M' + (mid + off - 4).toFixed(1) + ' ' + (yy + 5) + 'L'
            + (mid + off + 4).toFixed(1) + ' ' + (yy - 5),
          stroke: '#989285', 'stroke-width': '1.2', fill: 'none',
          'vector-effect': 'non-scaling-stroke' }));
      });
    });
    host.appendChild(el('text', { x: mid.toFixed(1), y: '11',
      'text-anchor': 'middle', fill: '#b3ac9e', 'font-size': '9',
      'font-family': 'IBM Plex Mono, monospace' })).textContent =
      b.days + ' days not recorded';
  });

  pads.forEach((p) => {
    const rows = S.series[p].series;
    let bars = '', pts = '';
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      const x = px(L.toX(r.epoch));
      if (x < -2 || x > box.w + 2) { continue; }
      const xs = x.toFixed(1);
      const hi = cy(r.mean + r.sd), lo = cy(Math.max(r.mean - r.sd, lim[0]));
      if (lo - hi > 0.7) { bars += 'M' + xs + ' ' + lo.toFixed(1) + 'L' + xs + ' ' + hi.toFixed(1); }
      pts += 'M' + xs + ' ' + cy(r.mean).toFixed(1) + 'L' + xs + ' ' + cy(r.mean).toFixed(1);
    }
    host.appendChild(el('path', { d: bars, fill: 'none', stroke: COLOUR[p],
      'stroke-width': '0.9', opacity: '0.4', 'vector-effect': 'non-scaling-stroke' }));
    host.appendChild(el('path', { d: pts, fill: 'none', stroke: COLOUR[p],
      'stroke-width': CAP[p] === 'square' ? '3' : '3.4',
      'stroke-linecap': CAP[p], opacity: '0.85',
      'vector-effect': 'non-scaling-stroke' }));
  });

  const hr = S.hover >= 0 ? L.rows[S.hover] : null;
  if (hr) {
    const x = px(L.toX(hr.epoch));
    if (x >= -2 && x <= box.w + 2) {
      host.appendChild(el('path', { d: 'M' + x.toFixed(1) + ' 0L' + x.toFixed(1)
        + ' ' + box.h, stroke: '#a4553a', 'stroke-width': '0.8', opacity: '0.5',
        fill: 'none', 'vector-effect': 'non-scaling-stroke' }));
      pads.forEach((p) => {
        const r = (S.series[p].series || [])[S.hover];
        if (!r) { return; }
        host.appendChild(el('path', {
          d: 'M' + x.toFixed(1) + ' ' + cy(r.mean).toFixed(1) + 'L'
            + x.toFixed(1) + ' ' + cy(r.mean).toFixed(1),
          fill: 'none', stroke: COLOUR[p], 'stroke-width': '8',
          'stroke-linecap': 'round', 'vector-effect': 'non-scaling-stroke' }));
      });
    }
  }

  axes('camp', box, [L.toEpoch(v[0]), L.toEpoch(v[1])], lim,
       { date: true, fy: (val) => val.toFixed(1) + ' pm' });
  ticks($('[data-yc="camp"]'), lim, 5, (v2) => fmt(v2));
  const xr = $('[data-xr="camp"]');
  xr.innerHTML = '';
  for (let i = 0; i <= 6; i++) {
    const t = L.toEpoch(v[0] + (v[1] - v[0]) * i / 6);
    const d = document.createElement('div');
    d.textContent = new Date(t * 1000).toLocaleDateString('en-GB',
      { day: '2-digit', month: 'short' });
    xr.appendChild(d);
  }
  legend('camp', pads);
  $('#camp-title').textContent = 'SNR of every passage';
  $('#camp-foot').textContent = L.rows.length + ' passages  ·  '
    + 'wheel or drag to zoom, shift-drag to pan, double-click for all';
  const read = $('#camp-read');
  read.innerHTML = '';
  const items = hr
    ? [['record', hr.file], ['axles', hr.axles === null ? '—' : String(hr.axles)],
       ['km/h', hr.speed_kmh == null ? '—' : hr.speed_kmh.toFixed(0)]].concat(
       pads.map((p) => {
         const r = (S.series[p].series || [])[S.hover];
         return ['PAD ' + p, r ? fmt(r.mean) + ' ± ' + fmt(r.sd) : '—'];
       }))
    : [['', 'move across to read a passage, click to open it']];
  items.forEach(([k, val]) => {
    const d = document.createElement('div');
    d.innerHTML = '<span>' + k + '</span><b>' + val + '</b>';
    read.appendChild(d);
  });
}

function drawHead() {
  const R = S.record;
  if (!R) { return; }
  const T = R.trace || {};
  $('#when').textContent = R.record.local.slice(0, 19);
  $('#ident').textContent = R.record.folder + '  ·  ' + R.record.file + '  ·  '
    + R.seconds.toFixed(1) + ' s  ·  ' + R.fs.toFixed(0) + ' Hz  ·  baseline '
    + R.spectra.base.toFixed(4) + ' nm, kept';
  const stats = [
    [String(T.axles === undefined ? '—' : T.axles), 'axles on PAD ' + R.pad, COLOUR[R.pad]],
    [(T.n_x === undefined ? '—' : T.n_x + ' / ' + T.n_d), 'X chain / D chain', '#1d1b17'],
    [(T.speed_kmh == null ? '—' : T.speed_kmh.toFixed(0)), 'km/h', '#1d1b17'],
    [String(T.trains === undefined ? '—' : T.trains), 'trains in the record', '#1d1b17']
  ];
  const host = $('#stats');
  host.innerHTML = '';
  stats.forEach(([v, k, c]) => {
    const d = document.createElement('div');
    d.className = 'stat';
    d.innerHTML = '<b style="color:' + c + '">' + v + '</b><span>' + k + '</span>';
    host.appendChild(d);
  });
  $('#psd-note').textContent = 'Welch 2048, peak ' + R.spectra.psd.peak.toFixed(2) + ' Hz';
}

let raf = null;
function draw() {
  if (raf) { return; }
  raf = setTimeout(() => {
    raf = null;
    const svg = $('[data-zoom="raw"]');
    if (svg) {
      const sel = (n) => {
        const s = $('[data-el="' + n + '-sel"]');
        const svg2 = $('[data-zoom="' + n + '"]');
        if (!s || !svg2) { return; }
        const d = S.drag;
        if (!d || d.name !== n || d.pan) {
          s.setAttribute('width', '0');
          return;
        }
        const b = svg2.getBoundingClientRect();
        const lo = Math.min(d.a, d.b), hi = Math.max(d.a, d.b);
        const ylo = Math.min(d.ay, d.by), yhi = Math.max(d.ay, d.by);
        const flat = (S.axes[n] || {}).flat;
        const tall = yhi - ylo >= 0.05 && !flat;
        s.setAttribute('x', (lo * b.width).toFixed(1));
        s.setAttribute('width', ((hi - lo) * b.width).toFixed(1));
        s.setAttribute('y', tall ? (ylo * b.height).toFixed(1) : '0');
        s.setAttribute('height', tall
          ? ((yhi - ylo) * b.height).toFixed(1)
          : String(Math.round(b.height)));
        svg2.appendChild(s);
      };
      if (S.record) {
        drawHead();
        drawRaw();
        drawDetection();
        drawSpectrum('psd', 1e-6);
        drawSpectrum('fft', 1e-3);
        drawModes();
      }
      drawCampaign();
      ['raw', 'det', 'psd', 'fft', 'emd', 'camp'].forEach(sel);
      ['raw', 'det', 'psd', 'fft', 'emd', 'camp'].forEach(drawOverlay);
    }
  }, 0);
}

/* ------------------------------------------------------------------ loading */

async function loadRaw(pad, win) {
  if (!S.record) { return; }
  const w = win || [0, S.record.seconds];
  const j = await api('/api/raw/' + S.id + '?pad=' + pad + '&t0=' + w[0]
    + '&t1=' + w[1] + '&base=' + (S.keepBase ? 'keep' : 'remove'));
  if (!j.error) { S.raw[pad] = j; draw(); }
}

async function loadTrace(pad, win) {
  if (!S.record) { return; }
  const w = win || S.record.window;
  const j = await api('/api/trace/' + S.id + '?engine=' + S.engine + '&pad=' + pad
    + '&t0=' + w[0] + '&t1=' + w[1]);
  if (!j.error) { S.det[pad] = j; draw(); }
}

async function loadSpectra(pad) {
  if (!S.record) { return; }
  if (pad === S.record.pad) {
    S.record.pads = S.record.pads || {};
    S.record.pads[pad] = { psd: S.record.spectra.psd, fft: S.record.spectra.fft,
                           emd: S.record.emd };
    return;
  }
  const j = await api('/api/record/' + S.id + '?engine=' + S.engine + '&pad=' + pad);
  if (j.error) { return; }
  S.record.pads = S.record.pads || {};
  S.record.pads[pad] = { psd: j.spectra.psd, fft: j.spectra.fft, emd: j.emd };
  draw();
}

async function loadSeries(pad) {
  const j = await api('/api/series?engine=' + S.engine + '&pad=' + pad
    + '&folder=' + encodeURIComponent(S.folder || ''));
  if (!j.error) { S.series[pad] = j; draw(); }
}

async function loadRecords() {
  const j = await api('/api/records?engine=' + S.engine + '&pad=' + S.primary
    + '&folder=' + encodeURIComponent(S.folder || '') + '&q=' + encodeURIComponent(S.q));
  S.records = j.records || [];
  $('#railfoot').textContent = (j.totals ? j.totals.records : 0) + ' records  ·  '
    + (j.totals ? j.totals.axles : 0) + ' axles';
  drawRows();
  if (S.records.length && !S.records.some((r) => r.id === S.id)) {
    selectRecord((S.records.find((r) => r.axles) || S.records[0]).id);
  }
}

async function selectRecord(id) {
  S.id = id;
  drawRows();
  const row = $('.row[data-id="' + id + '"]');
  if (row) { row.scrollIntoView({ block: 'nearest' }); }
  const j = await api('/api/record/' + id + '?engine=' + S.engine + '&pad=' + S.primary);
  if (j.error) { return; }
  S.record = j;
  S.record.pads = {};
  S.raw = {}; S.det = {};
  ['raw', 'det', 'psd', 'fft', 'emd'].forEach((k) => {
    delete S.view[k];
    delete S.viewY[k];
  });
  draw();
  S.pads.forEach((p) => { loadRaw(p, null); loadTrace(p, null); loadSpectra(p); });
}

function togglePad(p) {
  const i = S.pads.indexOf(p);
  if (i >= 0) {
    if (S.pads.length === 1) { return; }
    S.pads.splice(i, 1);
    delete S.raw[p]; delete S.det[p]; delete S.series[p];
    if (S.record && S.record.pads) { delete S.record.pads[p]; }
  } else {
    S.pads.push(p);
    S.pads.sort((a, b) => a - b);
    loadSeries(p);
    loadRaw(p, view('raw', fullOf('raw')));
    loadTrace(p, view('det', fullOf('det')));
    loadSpectra(p);
  }
  S.primary = S.pads[0];
  drawPads();
  draw();
}

function drawRows() {
  const host = $('#rows');
  host.innerHTML = '';
  S.records.forEach((r) => {
    const d = document.createElement('div');
    d.className = 'row' + (r.id === S.id ? ' on' : '');
    d.dataset.id = r.id;
    d.innerHTML = '<div class="dot' + (r.axles ? '' : ' none') + '"></div>'
      + '<div class="w">' + r.local.slice(5, 19) + '</div>'
      + '<div class="a">' + (r.axles === null ? '·' : r.axles) + '</div>'
      + '<div class="v">' + (r.speed_kmh == null ? '' : r.speed_kmh.toFixed(0)) + '</div>';
    d.onclick = () => selectRecord(r.id);
    host.appendChild(d);
  });
}

function drawPads() {
  const host = $('#pads');
  host.innerHTML = '';
  for (let p = 1; p <= 12; p++) {
    const b = document.createElement('button');
    b.textContent = String(p);
    const on = S.pads.includes(p);
    b.className = on ? 'on' : '';
    b.style.background = on
      ? (p > 6
          ? 'repeating-linear-gradient(90deg,' + COLOUR[p] + ' 0 4px,'
            + '#ffffff 4px 6px)'
          : COLOUR[p])
      : '';
    b.onclick = () => togglePad(p);
    host.appendChild(b);
  }
}

function setBig(name) {
  S.big = (S.big === name) ? null : name;
  const work = $('#work');
  work.classList.toggle('focused', !!S.big);
  $$('#work > .card').forEach((c) => c.classList.remove('big'));
  if (S.big) {
    const card = $('[data-panel="' + S.big + '"]');
    if (card) {
      if (card.parentElement.classList.contains('trio')) {
        work.appendChild(card);
      }
      card.classList.add('big');
    }
  } else {
    ['psd', 'fft', 'emd'].forEach((n) => {
      const c = $('[data-panel="' + n + '"]');
      if (c && !c.parentElement.classList.contains('trio')) {
        $('.trio').appendChild(c);
      }
    });
  }
  setTimeout(draw, 40);
}

function step(delta) {
  if (!S.records.length) { return; }
  const i = S.records.findIndex((r) => r.id === S.id);
  const j = clip((i < 0 ? 0 : i) + delta, 0, S.records.length - 1);
  if (S.records[j] && S.records[j].id !== S.id) { selectRecord(S.records[j].id); }
}

/* ------------------------------------------------------------------- folder */

function tidyPath(v) {
  return String(v || '').trim().replace(/^["']+|["']+$/g, '').trim();
}

let browseSeq = 0;

async function openBrowse(path) {
  const mine = ++browseSeq;
  const j = await api('/api/browse?path=' + encodeURIComponent(path || ''));
  if (mine !== browseSeq) { return; }
  if (j.error && !(j.entries || []).length) {
    $('#dinfo').textContent = j.error;
    $('#use').disabled = true;
    return;
  }
  S.browse = j;
  S.pickPath = j.records > 0 ? j.path : null;
  $('#here').value = j.path;
  const host = $('#listing');
  host.innerHTML = '';
  (j.entries || []).forEach((e) => {
    const d = document.createElement('div');
    d.className = 'entry';
    d.innerHTML = '<div class="n">' + e.name + '</div>'
      + '<div class="c' + (e.records ? '' : ' none') + '">'
      + (e.records ? e.records + ' records' : 'no records') + '</div>';
    d.onclick = () => {
      $$('.entry', host).forEach((x) => x.classList.remove('on'));
      d.classList.add('on');
      S.pickPath = e.path;
      updatePick(e);
    };
    d.ondblclick = () => openBrowse(e.path);
    host.appendChild(d);
  });
  updatePick(j.records > 0 ? { name: j.path, records: j.records } : null);
}

function updatePick(e) {
  const info = $('#dinfo'), use = $('#use');
  if (!e || !e.records) {
    info.textContent = e ? 'that folder holds no .txt records' : 'pick a folder';
    use.disabled = true;
  } else {
    info.textContent = e.records + ' records in ' + e.name;
    use.disabled = false;
  }
}

let pollTimer = null;
function poll() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    const p = await api('/api/progress');
    const b = $('#busy');
    if (p.running) {
      b.dataset.job = '1';
      b.textContent = p.stage + ' ' + p.folder + '  ' + p.done + '/' + p.total;
      poll();
    } else {
      delete b.dataset.job;
      b.textContent = '';
      if (p.stage === 'done') { toast('took in ' + p.folder + ': ' + p.message); boot(true); }
      else if (p.stage === 'failed') { toast(p.message); }
    }
  }, 800);
}

/* --------------------------------------------------------------------- boot */

async function boot(keep) {
  const setup = await api('/api/setup');
  const eh = $('#engines');
  eh.innerHTML = '';
  (setup.engines || []).forEach((e) => {
    const b = document.createElement('button');
    b.textContent = e.name;
    b.title = e.about;
    b.disabled = !e.ready;
    b.className = e.id === S.engine ? 'on' : '';
    b.onclick = () => {
      S.engine = e.id;
      $$('#engines button').forEach((x) => x.classList.toggle('on', x === b));
      S.pads.forEach(loadSeries);
      loadRecords();
      if (S.id) { selectRecord(S.id); }
    };
    eh.appendChild(b);
  });
  const fh = $('#folders');
  fh.innerHTML = '';
  const total = (setup.folders || []).reduce((a, f) => a + f.records, 0);
  [{ folder: null, records: total }].concat(setup.folders || []).forEach((f) => {
    const b = document.createElement('button');
    b.textContent = (f.folder || 'everything') + ' ' + f.records;
    b.className = f.folder === S.folder ? 'on' : '';
    b.onclick = () => {
      S.folder = f.folder;
      $$('#folders button').forEach((x) => x.classList.toggle('on', x === b));
      S.pads.forEach(loadSeries);
      loadRecords();
    };
    fh.appendChild(b);
  });
  if (keep) { S.pads.forEach(loadSeries); await loadRecords(); return; }

  drawPads();
  ['raw', 'det', 'psd', 'fft', 'emd', 'camp'].forEach((k) => {
    wire($('[data-zoom="' + k + '"]'), k);
    controls(k);
  });
  $('[data-zoom="camp"]').addEventListener('click', () => {
    if (S.markMode) { return; }
    if (S.hover >= 0) {
      const L = campLayout();
      if (L && L.rows[S.hover]) { selectRecord(L.rows[S.hover].id); }
    }
  });
  $$('#work .ph').forEach((h) => {
    h.ondblclick = (ev) => {
      if (ev.target.closest('.ctl')) { return; }
      const card = h.closest('[data-panel]');
      if (card) { setBig(card.dataset.panel); }
    };
  });
  $('#pick').onclick = () => {
    $('#veil').classList.add('on');
    openBrowse('');
    setTimeout(() => { const h = $('#here'); h.focus(); h.select(); }, 30);
  };
  const goPath = () => {
    const v = tidyPath($('#here').value);
    if (v) { openBrowse(v); }
  };
  $('#go').onclick = goPath;
  $('#here').addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') { ev.preventDefault(); goPath(); }
  });
  $('#here').addEventListener('focus', (ev) => ev.target.select());
  $('#here').addEventListener('paste', (ev) => {
    const t = (ev.clipboardData || window.clipboardData).getData('text');
    if (!t) { return; }
    ev.preventDefault();
    ev.target.value = tidyPath(t);
    goPath();
  });
  $('#close').onclick = () => $('#veil').classList.remove('on');
  $('#veil').onclick = (ev) => { if (ev.target.id === 'veil') { $('#veil').classList.remove('on'); } };
  $('#up').onclick = () => { if (S.browse) { openBrowse(S.browse.parent); } };
  $('#use').onclick = async () => {
    if (!S.pickPath) { return; }
    $('#veil').classList.remove('on');
    const j = await api('/api/ingest?path=' + encodeURIComponent(S.pickPath)
      + '&engine=' + S.engine + '&pad=' + S.primary);
    if (!j.error) { toast('taking in ' + j.started); poll(); }
  };
  let timer = null;
  $('#q').addEventListener('input', (ev) => {
    S.q = ev.target.value;
    clearTimeout(timer);
    timer = setTimeout(loadRecords, 220);
  });
  document.addEventListener('keydown', (ev) => {
    if (ev.target.tagName === 'INPUT') { return; }
    if (ev.key === 'ArrowDown') { ev.preventDefault(); step(1); }
    if (ev.key === 'ArrowUp') { ev.preventDefault(); step(-1); }
    if (ev.key === 'Escape') {
      if ($('#veil').classList.contains('on')) { $('#veil').classList.remove('on'); }
      else if (S.big) { setBig(S.big); }
    }
  });
  let rz = null;
  window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(draw, 120); });
  S.pads.forEach(loadSeries);
  await loadRecords();
  poll();
}

boot(false);
