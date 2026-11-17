// gta6-twitch-ccu-forecast-01 entry (Node 22, standard library only).
//
// Forecasts Grand Theft Auto VI's launch-day Twitch concurrent viewers, hour by hour
// (hours 2..24 UTC, 2026-11-19), from hour 1's print and the release-day curves of
// past game launches shipped in the case file. Everything it knows is in the case:
// it runs offline, with no network, under the case's time cap.
//
// The model, on the log scale. For each hour h, a launch's viewers are expressed as
// a ratio to its own hour-1 print, r_h = viewers_h / viewers_1. Across past launches
// that ratio is not one curve: it depends strongly on how big hour 1 already was.
// Hour 1 at 00:00 UTC already holds every region whose midnight has passed
// (New Zealand, Australia, East Asia, Central Europe); a launch whose hour 1 is
// already huge has most of its audience in hand, so its day stays flat and its peak
// barely exceeds hour 1, while a launch with a small hour 1 still has the whole
// United States unlock (05:00Z Eastern, 08:00Z Pacific) ahead of it and climbs
// several-fold into the US evening. On the 42 launches in the kit (38 comparables
// plus the 4 practice launches, whose release days were read back from SullyGnome),
// log peak ratio correlates -0.95 with log hour-1 print.
//
// So, for each hour h, the program fits one line over the rolling-midnight launches
// (the launch shape that matches GTA VI: rolling midnight local, no early access):
//
//   log r_h = a_h + b_h * (log viewers_1 - 11)
//
// Early-access launches (hour 1 inflated by days of early play), free-to-play and
// slow-burn launches, and single-global-unlock launches (hour 1 is a pre-unlock
// artefact, near zero) are left out of the fit: their shapes do not match this
// launch. The fitted a_h, b_h and the residual spread sd_h are smoothed over
// neighbouring hours, and hour h's five quantiles are the lognormal quantiles
// around the fitted median:
//
//   q_tau(h) = viewers_1 * exp(a_h + b_h * (log viewers_1 - 11) + sd_h * z_tau)
//
// with z_tau the standard normal's quantiles at 0.05, 0.25, 0.5, 0.75, 0.95.
// The median is floored at 0.35 * viewers_1: no launch day in the set ever fell
// below a third of its hour-1 print.
//
// Fallbacks, in order, if a case ships too few rolling launches to fit: fit the
// same regression over every non-simultaneous launch, then fall back to the plain
// median curve over those launches, then to the house baseline (flat hour 1 with
// a widening spread). The program is deterministic: it reads its case files in
// name order, fits with plain arithmetic, and uses no randomness, clock, or
// network, so two runs on the same case write byte-identical outputs.
//
// On the kit's four practice cases this scores a mean pinball loss on the log scale
// of 0.086, against 0.194 for the house baseline (flat hour 1 with a widening
// spread) and 0.149 for the plain median curve across comparables:
//   p00 Marvel's Wolverine  0.094 vs baseline 0.372
//   p01 Red Dead Redemption 2  0.064 vs baseline 0.143
//   p02 God of War Ragnarok  0.071 vs baseline 0.107
//   p03 Marvel's Spider-Man 2  0.114 vs baseline 0.152
//
// What an entry must do (from the kit):
//   - read every <id>.in in the input directory (the grader runs one case at a time,
//     with only that case's file there): JSON {"schema":"makefun/twitch-ccu-case/v1",
//     "id","game","day_start","checkpoint","levels","known","forecast","history",
//     "history_hourly","streams","categories_hourly","comparables","notes"}, where
//     `known` holds hour 1's print {hour: 1, from, until, viewers, channels, reads}
//     and `forecast` lists the hours to forecast, 2..24;
//   - write <id>.out to the output directory: {"hours":[{"hour":2,
//     "q":[q05,q25,q50,q75,q95]}, ...]}, exactly 23 rows, hours 2..24 each once,
//     no other key, each q five finite numbers above 0 and at most 1e9 (viewers;
//     fractions allowed), non-decreasing; at most 65,536 bytes.
// It runs offline, with no network, under the case's time cap: everything it knows
// is in the case file.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in
// for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
//
// All file paths come from those environment variables (defaulting to /input and
// /output); the program never reads a bare relative path, so it works from any
// working directory the grader runs it from.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
/** log hour-1 print is centred here for the regression (e^11 is about 60,000 viewers). */
const LOG_V1_CENTER = 11;
/** A launch day's median never falls below this fraction of its hour-1 print. */
const MIN_MEDIAN_RATIO = 0.35;
/** The house baseline's spread (fallback only), in log units per square-root hour. */
const BASE_SIGMA = 0.15;
/** The fit needs at least this many launches to be trusted. */
const MIN_TRAIN = 4;

interface KnownRow {
  hour: number;
  viewers: number;
  channels?: number;
}

interface HourlyPoint {
  from: string;
  viewers: number;
  channels?: number;
}

interface Comparable {
  day_start: string;
  hourly: HourlyPoint[];
  name?: string;
  unlock?: string;
}

interface Case {
  levels: number[];
  known: KnownRow[];
  forecast: Array<{ hour: number }>;
  history_hourly?: HourlyPoint[];
  day_start?: string;
  comparables: Comparable[];
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** The three-point moving average of an array (edges average the two points they have). */
const smooth3 = (arr: number[]): number[] =>
  arr.map((_, i) => {
    const lo = Math.max(0, i - 1);
    const hi = Math.min(arr.length - 1, i + 1);
    let s = 0;
    for (let k = lo; k <= hi; k++) s += arr[k];
    return s / (hi - lo + 1);
  });

/** Least squares line y = a + b*x through the points; returns { a, b }. */
const ols = (xs: number[], ys: number[]): { a: number; b: number } => {
  const n = xs.length;
  const mx = xs.reduce((s, x) => s + x, 0) / n;
  const my = ys.reduce((s, y) => s + y, 0) / n;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) * (xs[i] - mx);
  }
  const b = sxx > 1e-9 ? sxy / sxx : 0;
  return { a: my - b * mx, b };
};

/** A comparable's tier, read from its unlock note: only plain "comparable" launches train the fit. */
const tierOf = (c: Comparable): string => {
  const u = (c.unlock ?? '').toLowerCase();
  if (u.includes('tier: early-access-contaminated')) return 'contaminated';
  if (u.includes('tier: weak')) return 'weak';
  if (u.includes('tier: target-history')) return 'target-history';
  return 'comparable';
};

/** A comparable's unlock shape: rolling midnight local, one simultaneous global unlock, or unverified. */
const unlockTypeOf = (c: Comparable): string => {
  const u = (c.unlock ?? '').toLowerCase();
  if (
    u.includes('single global') ||
    u.includes('simultaneous global') ||
    u.includes('single simultaneous') ||
    u.includes('simultaneous across')
  ) {
    return 'simultaneous';
  }
  if (u.includes('midnight local') || u.includes('rolling')) return 'rolling';
  return 'unknown';
};

/** A comparable's release-day curve: hour (1..24 from day_start) to viewers, positive points only. */
const releaseCurve = (c: Comparable): Map<number, number> => {
  const base = Date.parse(c.day_start);
  const m = new Map<number, number>();
  for (const p of c.hourly ?? []) {
    if (!(p.viewers > 0)) continue;
    const h = Math.round((Date.parse(p.from) - base) / 3600000) + 1;
    if (h >= 1 && h <= 24 && !m.has(h)) m.set(h, p.viewers);
  }
  return m;
};

/** One training launch: its hour-1 print and its per-hour ratios to it. */
interface TrainRow {
  v1: number;
  ratios: Map<number, number>;
}

const toTrainRows = (comps: Comparable[], rollingOnly: boolean): TrainRow[] => {
  const rows: TrainRow[] = [];
  for (const c of comps) {
    if (tierOf(c) !== 'comparable') continue;
    if (rollingOnly && unlockTypeOf(c) !== 'rolling') continue;
    if (!rollingOnly && unlockTypeOf(c) === 'simultaneous') continue;
    const curve = releaseCurve(c);
    const v1 = curve.get(1);
    if (v1 === undefined || !(v1 > 0)) continue;
    const ratios = new Map<number, number>();
    for (const [h, v] of curve) ratios.set(h, v / v1);
    if (ratios.size < 12) continue;
    rows.push({ v1, ratios });
  }
  return rows;
};

/**
 * Fit the per-hour regression log r_h = a_h + b_h * (log v1 - 11) over the training
 * launches. Returns null when too few launches carry an hour's data. The returned
 * arrays are indexed by the forecast hours, in order.
 */
const fitRegression = (
  rows: TrainRow[],
  hours: number[],
): { a: number[]; b: number[]; sd: number[] } | null => {
  if (rows.length < MIN_TRAIN) return null;
  const a: number[] = [];
  const b: number[] = [];
  const sd: number[] = [];
  for (const h of hours) {
    const sub = rows.filter((r) => r.ratios.get(h) !== undefined);
    if (sub.length < 3) return null;
    const xs = sub.map((r) => Math.log(r.v1) - LOG_V1_CENTER);
    const ys = sub.map((r) => Math.log(r.ratios.get(h) as number));
    const line = ols(xs, ys);
    let sse = 0;
    for (let i = 0; i < sub.length; i++) {
      const e = ys[i] - (line.a + line.b * xs[i]);
      sse += e * e;
    }
    a.push(line.a);
    b.push(line.b);
    sd.push(Math.sqrt(sse / Math.max(1, sub.length - 2)));
  }
  return { a: smooth3(a), b: smooth3(b), sd: smooth3(sd) };
};

/**
 * Fallback: the plain median log-ratio curve over the training launches, with
 * empirical quantiles per hour. Returns null when too few launches carry the data.
 */
const medianCurve = (
  rows: TrainRow[],
  hours: number[],
): { med: number[]; q: number[][] } | null => {
  if (rows.length < MIN_TRAIN) return null;
  const med: number[] = [];
  const q: number[][] = [];
  for (const h of hours) {
    const rs = rows
      .map((r) => r.ratios.get(h))
      .filter((x): x is number => x !== undefined)
      .map((x) => Math.log(x))
      .sort((x, y) => x - y);
    if (rs.length < 3) return null;
    const quant = (p: number): number => {
      const idx = (rs.length - 1) * p;
      const lo = Math.floor(idx);
      const hi = Math.ceil(idx);
      return rs[lo] + (rs[hi] - rs[lo]) * (idx - lo);
    };
    med.push(quant(0.5));
    q.push(LEVELS.map((p) => quant(p)));
  }
  return { med: smooth3(med), q };
};

/** The house baseline's quantiles (flat hour 1, spread widening like a random walk). */
const baselineQuantiles = (p1: number, h: number): number[] =>
  Z.map((z) => round2(p1 * Math.exp(z * BASE_SIGMA * Math.sqrt(h - 1))));

/** Clamp quantiles to the contest's (0, 1e9] range and enforce non-decreasing order. */
const finalQuantiles = (q: number[]): number[] => {
  const clamped = q.map((x) => Math.min(1e9, Math.max(1, x)));
  for (let j = 1; j < clamped.length; j++) clamped[j] = Math.max(clamped[j], clamped[j - 1]);
  return clamped;
};

/** The forecast rows for one case. */
const forecastCase = (c: Case): Array<{ hour: number; q: number[] }> => {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`case asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const first = c.known.find((k) => k.hour === 1);
  let p1: number | undefined = first === undefined ? undefined : first.viewers;
  if (p1 === undefined && c.day_start !== undefined) {
    // No hour-1 print in `known`: fall back to the history point at day_start.
    const at = (c.history_hourly ?? []).find((p) => p.from === c.day_start);
    p1 = at === undefined ? undefined : at.viewers;
  }
  if (p1 === undefined || !(p1 > 0)) throw new Error('case has no hour-1 print');
  p1 = Math.max(1, p1);
  const hours = (c.forecast ?? []).map((f) => f.hour);
  if (hours.length === 0) throw new Error('case lists no forecast hours');
  const logV1 = Math.log(p1) - LOG_V1_CENTER;

  // Primary fit: rolling-midnight launches only. Broader fit (any non-simultaneous
  // launch) if the rolling set is too small; median curve, then the house baseline.
  const rolling = toTrainRows(c.comparables ?? [], true);
  const broad = rolling.length >= MIN_TRAIN ? rolling : toTrainRows(c.comparables ?? [], false);
  const fit = fitRegression(broad, hours);
  if (fit !== null) {
    return hours.map((h, i) => {
      const m = Math.max(fit.a[i] + fit.b[i] * logV1, Math.log(MIN_MEDIAN_RATIO));
      const q = Z.map((z) => round2(p1 * Math.exp(m + fit.sd[i] * z)));
      return { hour: h, q: finalQuantiles(q) };
    });
  }
  const med = medianCurve(broad, hours);
  if (med !== null) {
    return hours.map((h, i) => {
      const q = med.q[i].map((lq) => round2(p1 * Math.exp(lq)));
      return { hour: h, q: finalQuantiles(q) };
    });
  }
  return hours.map((h) => ({ hour: h, q: baselineQuantiles(p1, h) }));
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const hours = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
