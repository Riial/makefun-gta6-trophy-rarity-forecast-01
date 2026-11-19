// gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn rate
// (percent of tracked players) at the case's target, as five quantiles.
//
// Model, in one paragraph: a rate is a ratio, earned / players-tracked, and the
// two sides move for different reasons, so they are forecast separately. Both
// counts are fitted in log space against log time since release (recency
// weighted), extrapolated to the target, and divided back into a rate. Earned
// counts never fall and the story trophies nest inside each other, so those
// constraints are enforced on the extrapolation. The spread is honest about
// three things: a random walk over the horizon, the fit's own residual grown
// by how far past the data the target sits, and the refresh jumps Exophase
// makes when it reveals hours of hidden growth at once (estimated from the
// jumps the case's own history shows, with a cautious prior when it shows
// none). On a frozen page the refresh drop is a lower-tail risk only: the
// median stays at the trend centre (the launch-night drop came from one
// refresh while earned counts were still frozen; over days earned counts
// grow too), and only the 5%/25% quantiles reach down through the refresh
// mixture. The median is read at the expected last refresh before the target,
// not at the target itself, because the answering read shows the page as of
// its last refresh. Everything is deterministic: no randomness, no clock,
// no network, standard library only.
//
// Run: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Random-walk spread, logit units per square-root day (house baseline's scale). */
const WALK_SIGMA = 0.5;
/** Recency half-life for the trend fits, in hours. */
const HALF_LIFE_H = 24;
/** Default gap between Exophase refreshes when the history shows none, hours. */
const DEFAULT_REFRESH_GAP_H = 6;
/** Log player growth the first refresh revealed on launch night, ln(3868/638):
 * the page was frozen at 638 tracked players, then its read at 04:30 UTC showed
 * 3,868 (RULES.txt). Used only as the prior drop scale when a case's history
 * shows no refresh at all, so its trend cannot see the hidden growth. */
const LAUNCH_REFRESH_LOG_GROWTH = 1.8;
/** Spread clamp, logit units. */
const SPREAD_MIN = 0.12;
const SPREAD_MAX = 3.0;

/** Story chain, in play order: each step's earners are a subset of the last. */
const STORY_CHAIN = [2, 3, 4, 5, 6, 7];
/** Platinum needs every other trophy; 100% completion needs the rest. */
const PLATINUM_ID = 1;
const COMPLETION_ID = 14;

interface TrophyDef { id: number; }
interface HistoryRead {
  slot: string;
  players_tracked: number;
  trophies_earned?: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { release_date: string };
  trophies: TrophyDef[];
  history: HistoryRead[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logitOf = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const sigmoidPct = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Inverse standard normal CDF (Acklam's rational approximation). */
function normInv(p: number): number {
  if (p <= 0) return -8;
  if (p >= 1) return 8;
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const plow = 0.02425;
  const phigh = 1 - plow;
  let q: number;
  let r: number;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > phigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Error function (Abramowitz & Stegun 7.1.26), for the mixture CDF. */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
  return sign * y;
}
function normCdf(x: number, mu: number, sigma: number): number {
  return 0.5 * (1 + erf((x - mu) / (sigma * Math.SQRT2)));
}

/**
 * Quantile of the refresh mixture: with probability (1 - p) the rate sits
 * around `centre` with spread `s0`; with probability p a refresh lands in
 * the horizon and the rate sits `drop` logit units lower. Solved by bisection
 * on the mixture CDF. Used ONLY for lower-tail quantiles (see the quantile
 * step): the median is the centre by construction, so a near-certain refresh
 * (p close to 1 on a long horizon) can no longer drag the median itself onto
 * the dropped distribution.
 */
function mixtureQuantile(level: number, centre: number, s0: number, drop: number, p: number): number {
  let lo = centre - drop - 8 * s0 - 1;
  let hi = centre + 8 * s0 + 1;
  for (let k = 0; k < 80; k++) {
    const mid = (lo + hi) / 2;
    const cdf = (1 - p) * normCdf(mid, centre, s0) + p * normCdf(mid, centre - drop, s0);
    if (cdf < level) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

function mean(xs: number[]): number {
  return xs.reduce((s, x) => s + x, 0) / xs.length;
}
function std(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((s, x) => s + (x - m) * (x - m), 0) / (xs.length - 1));
}

/**
 * Weighted least squares of y against x. Returns {a, b, residStd} for
 * y ~= a + b*x, or null when there are too few points to fit.
 */
function wls(xs: number[], ys: number[], ws: number[]): { a: number; b: number; residStd: number } | null {
  if (xs.length < 2) return null;
  let sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    sw += ws[i];
    sx += ws[i] * xs[i];
    sy += ws[i] * ys[i];
    sxx += ws[i] * xs[i] * xs[i];
    sxy += ws[i] * xs[i] * ys[i];
  }
  const denom = sw * sxx - sx * sx;
  if (Math.abs(denom) < 1e-12) return null;
  const b = (sw * sxy - sx * sy) / denom;
  const a = (sy - b * sx) / sw;
  const resid = xs.map((x, i) => ys[i] - (a + b * x));
  return { a, b, residStd: std(resid) };
}

interface ReadRow {
  tH: number; // hours since release
  n: number; // players tracked
  rates: Array<number | null>;
}

/** Collapse consecutive identical reads (Exophase's stale page repeats itself). */
function dedupeReads(rows: ReadRow[]): ReadRow[] {
  const out: ReadRow[] = [];
  for (const r of rows) {
    const prev = out[out.length - 1];
    if (prev && prev.n === r.n && prev.rates.length === r.rates.length &&
        prev.rates.every((v, i) => v === r.rates[i])) continue;
    out.push(r);
  }
  return out;
}

function forecastCase(c: Case): { trophies: Array<{ id: number; q: number[] }> } {
  const releaseMs = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const targetMs = Date.parse(c.target);
  const tTargetH = Math.max(0, (targetMs - releaseMs) / 3_600_000);
  const horizonDays = Math.max(0, c.horizon_minutes / 1440);
  const horizonH = Math.max(0, c.horizon_minutes / 60);
  const nTrophies = c.trophies.length;

  const raw: ReadRow[] = (c.history ?? [])
    .map((h) => ({
      tH: Math.max(0, (Date.parse(h.slot) - releaseMs) / 3_600_000),
      n: h.players_tracked,
      rates: h.rates ?? [],
    }))
    .filter((r) => Number.isFinite(r.tH) && r.n > 0)
    .sort((a, b) => a.tH - b.tH);
  const reads = dedupeReads(raw);

  // Fallback when the case carries no usable history at all.
  if (reads.length === 0) {
    const s = 1.0;
    return {
      trophies: c.trophies.map((t) => ({
        id: t.id,
        q: c.levels.map((lv) => round4(Math.min(100, Math.max(0.01, sigmoidPct(logitOf(1) + normInv(lv) * s))))),
      })),
    };
  }

  const tLast = reads[reads.length - 1].tH;
  const xOf = (tH: number): number => Math.log(1 + tH);
  const weightOf = (tH: number): number => Math.pow(0.5, Math.max(0, tLast - tH) / HALF_LIFE_H);

  // --- Refresh events: consecutive distinct reads where the page moved. ---
  const refreshIdx: number[] = [];
  for (let j = 1; j < reads.length; j++) {
    const prev = reads[j - 1];
    const cur = reads[j];
    const nJump = Math.abs(cur.n - prev.n) / prev.n > 0.005;
    let rateMoved = false;
    for (let i = 0; i < nTrophies; i++) {
      const a = prev.rates[i];
      const b = cur.rates[i];
      if (a !== null && a !== undefined && b !== null && b !== undefined && Math.abs(b - a) > 0.02) {
        rateMoved = true;
        break;
      }
    }
    if (nJump || rateMoved) refreshIdx.push(j);
  }
  let refreshGapH = DEFAULT_REFRESH_GAP_H;
  if (refreshIdx.length >= 2) {
    const gaps: number[] = [];
    for (let k = 1; k < refreshIdx.length; k++) gaps.push(reads[refreshIdx[k]].tH - reads[refreshIdx[k - 1]].tH);
    gaps.sort((a, b) => a - b);
    refreshGapH = Math.max(1, gaps[Math.floor(gaps.length / 2)]);
  }
  const expectedJumps = Math.max(0, horizonH / refreshGapH);
  // Per-trophy logit jumps at refreshes, for the spread.
  const jumpStdByTrophy: number[] = [];
  for (let i = 0; i < nTrophies; i++) {
    const deltas: number[] = [];
    for (const j of refreshIdx) {
      const a = reads[j - 1].rates[i];
      const b = reads[j].rates[i];
      if (a !== null && a !== undefined && b !== null && b !== undefined) {
        deltas.push(logitOf(b) - logitOf(a));
      }
    }
    jumpStdByTrophy.push(std(deltas));
  }
  const observedJumpStds = jumpStdByTrophy.filter((s) => s > 0);
  const medianJumpStd = observedJumpStds.length > 0
    ? observedJumpStds.sort((a, b) => a - b)[Math.floor(observedJumpStds.length / 2)]
    : 0;

  // --- Players-tracked trend: log N against log time. ---
  const xsN: number[] = [];
  const ysN: number[] = [];
  const wsN: number[] = [];
  for (const r of reads) {
    xsN.push(xOf(r.tH));
    ysN.push(Math.log(r.n));
    wsN.push(weightOf(r.tH));
  }
  const fitN = wls(xsN, ysN, wsN);
  const lastN = reads[reads.length - 1].n;
  const logNTarget = fitN ? fitN.a + fitN.b * xOf(tTargetH) : Math.log(lastN);
  const nTarget = Math.max(lastN, Math.exp(logNTarget));
  const residN = fitN ? fitN.residStd : 0;

  // The answering read shows the page as of its last refresh, so the median
  // is read a half refresh-gap before the target (flat trends: no effect).
  const tEvalH = Math.max(0, tTargetH - refreshGapH / 2);
  const xEval = xOf(tEvalH);
  const xTarget = xOf(tTargetH);
  const extrapFactor = Math.max(0, xTarget / Math.max(1e-9, xOf(tLast)) - 1);

  const lastRates: Array<number | null> = reads[reads.length - 1].rates;
  const medians: number[] = [];
  const spreads: number[] = [];
  const baseSigmas: number[] = [];
  const refreshDrops: number[] = [];
  const earnedLast: number[] = [];
  const earnedPred: number[] = [];
  // A page that never moved in its history hides its growth: the trend is
  // flat because Exophase is frozen, not because nothing is happening. For
  // those cases the lower tail is a refresh mixture (see the quantile step),
  // not a symmetric spread.
  const frozenPage = refreshIdx.length === 0;
  const pRefresh = 1 - Math.exp(-horizonH / DEFAULT_REFRESH_GAP_H);

  for (let i = 0; i < nTrophies; i++) {
    // Earned counts, with a continuity correction where the rate is zero.
    const pts: Array<{ x: number; y: number; w: number }> = [];
    for (const r of reads) {
      const rate = r.rates[i];
      if (rate === null || rate === undefined) continue;
      const e = Math.max(0.5, (rate / 100) * r.n);
      pts.push({ x: xOf(r.tH), y: Math.log(e), w: weightOf(r.tH) });
    }
    const lastRate = lastRates[i];
    const lastE = lastRate === null || lastRate === undefined
      ? 0.5
      : Math.max(0.5, (lastRate / 100) * lastN);
    earnedLast.push(lastE);

    let ePred: number;
    let residE = 0;
    if (pts.length >= 2) {
      const fit = wls(pts.map((p) => p.x), pts.map((p) => p.y), pts.map((p) => p.w));
      if (fit) {
        ePred = Math.exp(fit.a + fit.b * xEval);
        residE = fit.residStd;
      } else {
        ePred = lastE;
      }
    } else {
      ePred = lastE;
    }
    // Earned counts never fall; nobody can earn more than there are players.
    ePred = Math.min(nTarget, Math.max(lastE, ePred));
    earnedPred.push(ePred);

    const median = (100 * ePred) / nTarget;
    medians.push(median);

    // --- Spread: random walk + fit uncertainty + refresh jumps. ---
    // The symmetric spread below is for histories that show refreshes, where
    // the jumps are already in the trend's noise. A frozen page gets the
    // refresh-mixture treatment at the quantile step instead.
    const walk = WALK_SIGMA * Math.sqrt(horizonDays);
    const fitUnc = Math.sqrt(residE * residE + residN * residN) * (1 + extrapFactor);
    const baseSigma = Math.sqrt(walk * walk + fitUnc * fitUnc);
    baseSigmas.push(baseSigma);
    // Expected logit drop if a refresh lands inside the horizon: the revealed
    // player growth, discounted by how fast this trophy's earners grow with
    // the player base (a 100% rate cannot drop at all). Only used frozen.
    const exposure = 1 - Math.sqrt(Math.min(1, (lastRate ?? medians[medians.length - 1] ?? 0) / 100));
    refreshDrops.push(frozenPage ? LAUNCH_REFRESH_LOG_GROWTH * exposure : 0);
    let jumpSigma = jumpStdByTrophy[i];
    if (!(jumpSigma > 0) && refreshIdx.length > 0) jumpSigma = medianJumpStd;
    const jump = (jumpSigma > 0 ? jumpSigma : 0) * Math.sqrt(expectedJumps);
    let s = Math.sqrt(baseSigma * baseSigma + jump * jump);
    // A trophy every tracked player holds in every read is structurally 100%.
    const pinnedFull = reads.every((r) => r.rates[i] === 100);
    if (pinnedFull) s = Math.min(s, 0.15);
    spreads.push(Math.min(SPREAD_MAX, Math.max(SPREAD_MIN, s)));
  }

  // --- Cross-trophy constraints on the predicted earned counts. ---
  const byId = new Map<number, number>();
  c.trophies.forEach((t, i) => byId.set(t.id, i));
  const applyChain = (ids: number[]): void => {
    let cap = Infinity;
    for (const id of ids) {
      const idx = byId.get(id);
      if (idx === undefined) continue;
      cap = Math.min(cap, earnedPred[idx]);
      earnedPred[idx] = cap;
    }
  };
  applyChain(STORY_CHAIN);
  const platIdx = byId.get(PLATINUM_ID);
  const compIdx = byId.get(COMPLETION_ID);
  if (platIdx !== undefined && compIdx !== undefined) {
    earnedPred[compIdx] = Math.min(earnedPred[compIdx], ...earnedPred.filter((_, i) => i !== platIdx && i !== compIdx));
    earnedPred[platIdx] = Math.min(earnedPred[platIdx], earnedPred[compIdx]);
  } else if (platIdx !== undefined) {
    earnedPred[platIdx] = Math.min(earnedPred[platIdx], ...earnedPred.filter((_, i) => i !== platIdx));
  }
  for (let i = 0; i < nTrophies; i++) {
    medians[i] = (100 * earnedPred[i]) / nTarget;
  }

  const zLevels = c.levels.map((lv) => normInv(lv));
  const trophies = c.trophies.map((t, i) => {
    const centre = logitOf(medians[i]);
    const s = spreads[i];
    const qs = c.levels.map((lv, k) => {
      // Frozen page: the refresh drop is a LOWER-TAIL risk, not the median.
      // The launch-night drop (1.8 logit) came from ONE refresh over ~4.5h
      // while earned counts were still frozen. Over a 4.5-day horizon earned
      // counts grow as players play, so the full drop is the tail case, not
      // the central one: with pRefresh near 1 the old mixture put the median
      // itself on the dropped distribution and under-forecast rare trophies.
      // So the median stays at the centre, the upper tail stays symmetric,
      // and only the lower tail reaches down through the mixture.
      let l: number;
      if (!frozenPage) {
        l = centre + zLevels[k] * s;
      } else if (lv < 0.5) {
        l = mixtureQuantile(lv, centre, Math.max(SPREAD_MIN, baseSigmas[i]), refreshDrops[i], pRefresh);
      } else if (lv === 0.5) {
        l = centre;
      } else {
        l = centre + zLevels[k] * s;
      }
      const v = sigmoidPct(l);
      return round4(Math.min(100, Math.max(0.01, v)));
    });
    // Rounding can break monotonicity by a hair; restore it.
    for (let k = 1; k < qs.length; k++) qs[k] = Math.max(qs[k], qs[k - 1]);
    return { id: t.id, q: qs };
  });
  return { trophies };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const out = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
