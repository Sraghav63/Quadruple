/** Small dense Float64 matrices. Partial pivoting matches the reference solves. */
export type Matrix = Float64Array[];
export const zeros = (r: number, c = r): Matrix => Array.from({ length: r }, () => new Float64Array(c));
export function eye(n: number, scale = 1): Matrix {
  const a = zeros(n); for (let i = 0; i < n; i++) a[i][i] = scale; return a;
}
export const copy = (a: Matrix): Matrix => a.map(row => row.slice());
export function transpose(a: Matrix): Matrix {
  const b = zeros(a[0].length, a.length);
  for (let i = 0; i < a.length; i++) for (let j = 0; j < b.length; j++) b[j][i] = a[i][j];
  return b;
}
export function mul(a: Matrix, b: Matrix): Matrix {
  const c = zeros(a.length, b[0].length);
  for (let i = 0; i < a.length; i++) for (let k = 0; k < b.length; k++) {
    const v = a[i][k]; if (v === 0) continue;
    for (let j = 0; j < c[0].length; j++) c[i][j] += v * b[k][j];
  }
  return c;
}
export function add(a: Matrix, b: Matrix, scale = 1): Matrix {
  return a.map((row, i) => row.map((v, j) => v + scale * b[i][j]));
}
export const scaled = (a: Matrix, s: number): Matrix => a.map(row => row.map(v => v * s));
export function solve(a: Matrix, b: Matrix): Matrix {
  const m = copy(a), x = copy(b), n = m.length, cols = b[0].length;
  for (let k = 0; k < n; k++) {
    let p = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(m[i][k]) > Math.abs(m[p][k])) p = i;
    if (!Number.isFinite(m[p][k]) || Math.abs(m[p][k]) < 1e-30) throw new Error('Singular design matrix');
    [m[k], m[p]] = [m[p], m[k]]; [x[k], x[p]] = [x[p], x[k]];
    for (let i = k + 1; i < n; i++) {
      const f = m[i][k] / m[k][k];
      for (let j = k + 1; j < n; j++) m[i][j] -= f * m[k][j];
      for (let j = 0; j < cols; j++) x[i][j] -= f * x[k][j];
    }
  }
  for (let i = n - 1; i >= 0; i--) for (let j = 0; j < cols; j++) {
    for (let k = i + 1; k < n; k++) x[i][j] -= m[i][k] * x[k][j];
    x[i][j] /= m[i][i];
  }
  return x;
}
export const sym = (a: Matrix): Matrix => scaled(add(a, transpose(a)), 0.5);
export const maxAbs = (a: Matrix): number => Math.max(...a.map(row => row.reduce((m, v) => Math.max(m, Math.abs(v)), 0)));

/** Higham's order-13 Padé approximant with 1-norm scaling and squaring. */
export function expm(a: Matrix): Matrix {
  const n = a.length;
  let norm = 0;
  for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < n; i++) s += Math.abs(a[i][j]); norm = Math.max(norm, s); }
  const s = Math.max(0, Math.ceil(Math.log2(norm / 5.371920351148152)));
  const x = scaled(a, 2 ** -s), i = eye(n), a2 = mul(x, x), a4 = mul(a2, a2), a6 = mul(a4, a2);
  const b = [64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800,
    129060195264000, 10559470521600, 670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1];
  const u = mul(x, add(add(add(mul(a6, add(add(scaled(a6, b[13]), scaled(a4, b[11])), scaled(a2, b[9]))), scaled(a6, b[7])), scaled(a4, b[5])), add(scaled(a2, b[3]), scaled(i, b[1]))));
  const v = add(add(add(mul(a6, add(add(scaled(a6, b[12]), scaled(a4, b[10])), scaled(a2, b[8]))), scaled(a6, b[6])), scaled(a4, b[4])), add(scaled(a2, b[2]), scaled(i, b[0])));
  let result = solve(add(v, u, -1), add(v, u));
  for (let k = 0; k < s; k++) result = mul(result, result);
  return result;
}

/** Structure-preserving doubling, in the same operation order as Python. */
export function dareDoubling(a: Matrix, b: Matrix, q: Matrix, r: Matrix): Matrix {
  let g = mul(b, solve(r, transpose(b))), h = copy(q);
  const i = eye(a.length);
  for (let k = 0; k < 100; k++) {
    const w = add(i, mul(g, h)), at = transpose(a);
    const aw = transpose(solve(transpose(w), at));
    const gn = add(g, mul(mul(aw, g), at));
    const hn = add(h, mul(mul(at, h), solve(w, a)));
    a = mul(aw, a);
    const done = maxAbs(add(hn, h, -1)) <= 1e-12 * Math.max(1, maxAbs(hn));
    g = sym(gn); h = sym(hn);
    if (done) break;
  }
  return h;
}
