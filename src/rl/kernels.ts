// WGSL kernels for training the nanoGPT ladder models in the browser (forward with stored
// activations, backward, AdamW) and for batched rollouts with a KV cache. All f32.
//
// Row layout: activations are [R, ...] with R = B * T, row r = b * T + t (sequence b, position t).
// KV caches are [B, CTX, d]: the k/v of sequence b at absolute position p live at row b * CTX + p.

const EPS = "0.000001";

const ERF = /* wgsl */ `
fn erf_(v: f32) -> f32 {
  let z = abs(v);
  let t = 1.0 / (1.0 + 0.5 * z);
  let r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
          t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return select(r - 1.0, 1.0 - r, v >= 0.0);
}`;

/**
 * C[M, N] = alpha * A @ B + beta * C, with A(m, k) = A[aOff + m*sam + k*sak] and B(k, n) = B[bOff + k*sbk + n*sbn].
 * C is row-major with row stride N starting at cOff. 64x64 tiles, 4x4 outputs per thread.
 */
export const GEMM = /* wgsl */ `
struct P { M: u32, N: u32, K: u32, sam: u32, sak: u32, sbk: u32, sbn: u32, aOff: u32, bOff: u32, cOff: u32, alpha: f32, beta: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
var<workgroup> As: array<f32, 1024>; // [16 k][64 m]
var<workgroup> Bs: array<f32, 1024>; // [16 k][64 n]
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let m0 = wg.y * 64u;
  let n0 = wg.x * 64u;
  let tid = lid.y * 16u + lid.x;
  var acc: array<f32, 16>;
  for (var k0 = 0u; k0 < p.K; k0 += 16u) {
    for (var e = 0u; e < 4u; e++) {
      let idx = tid + e * 256u;
      // pick the thread->element mapping that makes global loads contiguous
      var mm: u32; var ka: u32;
      if (p.sak == 1u) { mm = idx / 16u; ka = idx % 16u; } else { ka = idx / 64u; mm = idx % 64u; }
      let gm = m0 + mm; let gka = k0 + ka;
      var av = 0.0;
      if (gm < p.M && gka < p.K) { av = A[p.aOff + gm * p.sam + gka * p.sak]; }
      As[ka * 64u + mm] = av;
      var nn: u32; var kb: u32;
      if (p.sbn == 1u) { kb = idx / 64u; nn = idx % 64u; } else { nn = idx / 16u; kb = idx % 16u; }
      let gn = n0 + nn; let gkb = k0 + kb;
      var bv = 0.0;
      if (gn < p.N && gkb < p.K) { bv = B[p.bOff + gkb * p.sbk + gn * p.sbn]; }
      Bs[kb * 64u + nn] = bv;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      let ao = kk * 64u + lid.y * 4u;
      let bo = kk * 64u + lid.x * 4u;
      let a = vec4<f32>(As[ao], As[ao + 1u], As[ao + 2u], As[ao + 3u]);
      let b = vec4<f32>(Bs[bo], Bs[bo + 1u], Bs[bo + 2u], Bs[bo + 3u]);
      for (var i = 0u; i < 4u; i++) {
        acc[i * 4u] += a[i] * b.x;
        acc[i * 4u + 1u] += a[i] * b.y;
        acc[i * 4u + 2u] += a[i] * b.z;
        acc[i * 4u + 3u] += a[i] * b.w;
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let gm = m0 + lid.y * 4u + i;
    if (gm >= p.M) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let gn = n0 + lid.x * 4u + j;
      if (gn >= p.N) { continue; }
      let ci = p.cOff + gm * p.N + gn;
      var v = p.alpha * acc[i * 4u + j];
      if (p.beta != 0.0) { v += p.beta * C[ci]; }
      C[ci] = v;
    }
  }
}`;

export const EMBED = /* wgsl */ `
struct P { d: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> tok: array<u32>;
@group(0) @binding(2) var<storage, read> wte: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wg.y * 65535u + wg.x;
  for (var c = lid.x; c < p.d; c += 256u) { x[r * p.d + c] = wte[tok[r] * p.d + c]; }
}`;

export const RMSNORM = /* wgsl */ `
struct P { d: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let o = (wg.y * 65535u + wg.x) * p.d;
  var s = 0.0;
  for (var i = lid.x; i < p.d; i += 256u) { let v = x[o + i]; s += v * v; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var off = 128u; off > 0u; off >>= 1u) {
    if (lid.x < off) { red[lid.x] += red[lid.x + off]; }
    workgroupBarrier();
  }
  let r = inverseSqrt(red[0] / f32(p.d) + ${EPS});
  for (var i = lid.x; i < p.d; i += 256u) { y[o + i] = x[o + i] * r; }
}`;

/** dx = r*dy - x * r^3 * (dy.x) / d  (RMSNorm without weight); accumulate: dx += ... */
export const RMSNORM_BWD = /* wgsl */ `
struct P { d: u32, accumulate: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
var<workgroup> r1: array<f32, 256>;
var<workgroup> r2: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let o = (wg.y * 65535u + wg.x) * p.d;
  var ss = 0.0; var sd = 0.0;
  for (var i = lid.x; i < p.d; i += 256u) { let v = x[o + i]; ss += v * v; sd += v * dy[o + i]; }
  r1[lid.x] = ss; r2[lid.x] = sd;
  workgroupBarrier();
  for (var off = 128u; off > 0u; off >>= 1u) {
    if (lid.x < off) { r1[lid.x] += r1[lid.x + off]; r2[lid.x] += r2[lid.x + off]; }
    workgroupBarrier();
  }
  let r = inverseSqrt(r1[0] / f32(p.d) + ${EPS});
  let k = r * r * r * r2[0] / f32(p.d);
  for (var i = lid.x; i < p.d; i += 256u) {
    var v = r * dy[o + i] - x[o + i] * k;
    if (p.accumulate == 1u) { v += dx[o + i]; }
    dx[o + i] = v;
  }
}`;

/** Rotary on q (in place) and k; k and v go into the KV cache. Rows r = b*T + t at position start + t. */
export const ROPE = /* wgsl */ `
struct P { R: u32, T: u32, start: u32, d: u32, nHead: u32, half: u32, ctx: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> kc: array<f32>;
@group(0) @binding(3) var<storage, read_write> vc: array<f32>;
@group(0) @binding(4) var<storage, read> cs: array<vec2<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let n = p.R * p.nHead * p.half;
  for (var id = g.x; id < n; id += nw.x * 256u) {
    let i = id % p.half;
    let h = (id / p.half) % p.nHead;
    let r = id / (p.half * p.nHead);
    let b = r / p.T;
    let pos = p.start + r % p.T;
    let c = cs[pos * p.half + i];
    let hd = 2u * p.half;
    let qo = r * 3u * p.d + h * hd;
    let ko = qo + p.d;
    let vo = qo + 2u * p.d;
    let co = (b * p.ctx + pos) * p.d + h * hd;
    let q1 = qkv[qo + i]; let q2 = qkv[qo + p.half + i];
    qkv[qo + i] = q1 * c.x + q2 * c.y;
    qkv[qo + p.half + i] = -q1 * c.y + q2 * c.x;
    let k1 = qkv[ko + i]; let k2 = qkv[ko + p.half + i];
    kc[co + i] = k1 * c.x + k2 * c.y;
    kc[co + p.half + i] = -k1 * c.y + k2 * c.x;
    vc[co + i] = qkv[vo + i];
    vc[co + p.half + i] = qkv[vo + p.half + i];
  }
}`;

/** Inverse rotation of dq, dk (in place in dqkv), training layout (start = 0). */
export const ROPE_BWD = /* wgsl */ `
struct P { R: u32, T: u32, d: u32, nHead: u32, half: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> dqkv: array<f32>;
@group(0) @binding(2) var<storage, read> cs: array<vec2<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let n = p.R * p.nHead * p.half * 2u;
  for (var id = g.x; id < n; id += nw.x * 256u) {
    let i = id % p.half;
    let h = (id / p.half) % p.nHead;
    let which = (id / (p.half * p.nHead)) % 2u; // 0 = q, 1 = k
    let r = id / (p.half * p.nHead * 2u);
    let pos = r % p.T;
    let c = cs[pos * p.half + i];
    let o = r * 3u * p.d + which * p.d + h * 2u * p.half;
    let d1 = dqkv[o + i]; let d2 = dqkv[o + p.half + i];
    dqkv[o + i] = d1 * c.x - d2 * c.y;
    dqkv[o + p.half + i] = d1 * c.y + d2 * c.x;
  }
}`;

/** One workgroup per (head, t, b): causal attention over the cache. Also stores the softmax log-sum-exp. */
export const ATTN = (ctx: number) => /* wgsl */ `
struct P { T: u32, start: u32, d: u32, headDim: u32, nHead: u32, ctx: u32, scale: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> kc: array<f32>;
@group(0) @binding(3) var<storage, read> vc: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read_write> lse: array<f32>;
var<workgroup> sc: array<f32, ${ctx}>;
var<workgroup> q: array<f32, 64>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x; let t = wg.y; let b = wg.z;
  let r = b * p.T + t;
  let n = p.start + t + 1u;
  let hoff = h * p.headDim;
  let kbase = b * p.ctx * p.d + hoff;
  if (lid.x < p.headDim) { q[lid.x] = qkv[r * 3u * p.d + hoff + lid.x]; }
  workgroupBarrier();
  var m = -3.0e38;
  for (var s = lid.x; s < n; s += 64u) {
    let ko = kbase + s * p.d;
    var dot = 0.0;
    for (var i = 0u; i < p.headDim; i++) { dot += q[i] * kc[ko + i]; }
    dot *= p.scale;
    sc[s] = dot;
    m = max(m, dot);
  }
  red[lid.x] = m;
  workgroupBarrier();
  for (var off = 32u; off > 0u; off >>= 1u) {
    if (lid.x < off) { red[lid.x] = max(red[lid.x], red[lid.x + off]); }
    workgroupBarrier();
  }
  m = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var s = lid.x; s < n; s += 64u) { let e = exp(sc[s] - m); sc[s] = e; sum += e; }
  red[lid.x] = sum;
  workgroupBarrier();
  for (var off = 32u; off > 0u; off >>= 1u) {
    if (lid.x < off) { red[lid.x] += red[lid.x + off]; }
    workgroupBarrier();
  }
  let tot = red[0];
  if (lid.x == 0u) { lse[r * p.nHead + h] = m + log(tot); }
  if (lid.x < p.headDim) {
    var acc = 0.0;
    for (var s = 0u; s < n; s++) { acc += sc[s] * vc[kbase + s * p.d + lid.x]; }
    y[r * p.d + hoff + lid.x] = acc / tot;
  }
}`;

/** D[r, h] = dY[r, h, :] . Y[r, h, :]  (softmax backward helper) */
export const ATTN_D = /* wgsl */ `
struct P { R: u32, d: u32, headDim: u32, nHead: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> D: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  for (var id = g.x; id < p.R * p.nHead; id += nw.x * 256u) {
    let o = (id / p.nHead) * p.d + (id % p.nHead) * p.headDim;
    var s = 0.0;
    for (var i = 0u; i < p.headDim; i++) { s += dy[o + i] * y[o + i]; }
    D[id] = s;
  }
}`;

/** dQ for one (head, query t, b): recompute p_ts = exp(q.k*scale - lse), ds = p * (dY.v - D). */
export const ATTN_DQ = (ctx: number) => /* wgsl */ `
struct P { T: u32, d: u32, headDim: u32, nHead: u32, ctx: u32, scale: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> kc: array<f32>;
@group(0) @binding(3) var<storage, read> vc: array<f32>;
@group(0) @binding(4) var<storage, read> dy: array<f32>;
@group(0) @binding(5) var<storage, read> lse: array<f32>;
@group(0) @binding(6) var<storage, read> D: array<f32>;
@group(0) @binding(7) var<storage, read_write> dqkv: array<f32>;
var<workgroup> ds: array<f32, ${ctx}>;
var<workgroup> q: array<f32, 64>;
var<workgroup> g: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x; let t = wg.y; let b = wg.z;
  let r = b * p.T + t;
  let hoff = h * p.headDim;
  let kbase = b * p.ctx * p.d + hoff;
  if (lid.x < p.headDim) {
    q[lid.x] = qkv[r * 3u * p.d + hoff + lid.x];
    g[lid.x] = dy[r * p.d + hoff + lid.x];
  }
  workgroupBarrier();
  let l = lse[r * p.nHead + h];
  let Dr = D[r * p.nHead + h];
  for (var s = lid.x; s <= t; s += 64u) {
    let ko = kbase + s * p.d;
    var qk = 0.0; var gv = 0.0;
    for (var i = 0u; i < p.headDim; i++) { qk += q[i] * kc[ko + i]; gv += g[i] * vc[ko + i]; }
    let pr = exp(qk * p.scale - l);
    ds[s] = pr * (gv - Dr);
  }
  workgroupBarrier();
  if (lid.x < p.headDim) {
    var acc = 0.0;
    for (var s = 0u; s <= t; s++) { acc += ds[s] * kc[kbase + s * p.d + lid.x]; }
    dqkv[r * 3u * p.d + hoff + lid.x] = acc * p.scale;
  }
}`;

/** dK, dV for one (head, key s, b): sum over queries t >= s. */
export const ATTN_DKV = (ctx: number) => /* wgsl */ `
struct P { T: u32, d: u32, headDim: u32, nHead: u32, ctx: u32, scale: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> kc: array<f32>;
@group(0) @binding(3) var<storage, read> vc: array<f32>;
@group(0) @binding(4) var<storage, read> dy: array<f32>;
@group(0) @binding(5) var<storage, read> lse: array<f32>;
@group(0) @binding(6) var<storage, read> D: array<f32>;
@group(0) @binding(7) var<storage, read_write> dqkv: array<f32>;
var<workgroup> pr: array<f32, ${ctx}>;
var<workgroup> ds: array<f32, ${ctx}>;
var<workgroup> k: array<f32, 64>;
var<workgroup> v: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x; let s = wg.y; let b = wg.z;
  let hoff = h * p.headDim;
  let co = (b * p.ctx + s) * p.d + hoff;
  if (lid.x < p.headDim) { k[lid.x] = kc[co + lid.x]; v[lid.x] = vc[co + lid.x]; }
  workgroupBarrier();
  for (var t = s + lid.x; t < p.T; t += 64u) {
    let r = b * p.T + t;
    let qo = r * 3u * p.d + hoff;
    let go = r * p.d + hoff;
    var qk = 0.0; var gv = 0.0;
    for (var i = 0u; i < p.headDim; i++) { qk += qkv[qo + i] * k[i]; gv += dy[go + i] * v[i]; }
    let e = exp(qk * p.scale - lse[r * p.nHead + h]);
    pr[t] = e;
    ds[t] = e * (gv - D[r * p.nHead + h]);
  }
  workgroupBarrier();
  if (lid.x < p.headDim) {
    var dk = 0.0; var dv = 0.0;
    for (var t = s; t < p.T; t++) {
      let r = b * p.T + t;
      dk += ds[t] * qkv[r * 3u * p.d + hoff + lid.x];
      dv += pr[t] * dy[r * p.d + hoff + lid.x];
    }
    let ro = (b * p.T + s) * 3u * p.d + hoff + lid.x;
    dqkv[ro + p.d] = dk * p.scale;
    dqkv[ro + 2u * p.d] = dv;
  }
}`;

/** out = x + a * y over n elements */
export const AXPY = /* wgsl */ `
struct P { n: u32, a: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  for (var i = g.x; i < p.n; i += nw.x * 256u) { o[i] = x[i] + p.a * y[i]; }
}`;

export const GELU = /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> f: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
${ERF}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  for (var i = g.x; i < p.n; i += nw.x * 256u) { let v = f[i]; o[i] = 0.5 * v * (1.0 + erf_(v * 0.70710678118654752)); }
}`;

/** dg *= gelu'(f) in place */
export const GELU_BWD = /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> f: array<f32>;
@group(0) @binding(2) var<storage, read_write> dg: array<f32>;
${ERF}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  for (var i = g.x; i < p.n; i += nw.x * 256u) {
    let v = f[i];
    let cdf = 0.5 * (1.0 + erf_(v * 0.70710678118654752));
    let pdf = exp(-0.5 * v * v) * 0.3989422804014327;
    dg[i] *= cdf + v * pdf;
  }
}`;

/** dst[i] = src[idx[i]] (rows of width d) */
export const GATHER = /* wgsl */ `
struct P { d: u32, n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = wg.y * 65535u + wg.x;
  if (i >= p.n) { return; }
  for (var c = lid.x; c < p.d; c += 256u) { dst[i * p.d + c] = src[idx[i] * p.d + c]; }
}`;

/** dst[idx[i]] = src[i] (rows of width d; idx entries unique) */
export const SCATTER = /* wgsl */ `
struct P { d: u32, n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = wg.y * 65535u + wg.x;
  if (i >= p.n) { return; }
  for (var c = lid.x; c < p.d; c += 256u) { dst[idx[i] * p.d + c] = src[i * p.d + c]; }
}`;

/**
 * Policy-gradient / cross-entropy head, one workgroup per row of a logits chunk, in place:
 * dlogits = coef * (p - onehot(target)) + ent * p * (log p + H)   (the second term is d(-ent * H)/dlogits,
 * i.e. an entropy bonus); stats = [log p(target), entropy H].
 */
export const XENT = /* wgsl */ `
struct P { V: u32, rowOff: u32, ent: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> logits: array<f32>;
@group(0) @binding(2) var<storage, read> tgt: array<u32>;
@group(0) @binding(3) var<storage, read> coef: array<f32>;
@group(0) @binding(4) var<storage, read_write> stats: array<f32>;
var<workgroup> red: array<f32, 256>;
var<workgroup> red2: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = wg.x;
  let gi = p.rowOff + i;
  let o = i * p.V;
  var m = -3.0e38;
  for (var j = lid.x; j < p.V; j += 256u) { m = max(m, logits[o + j]); }
  red[lid.x] = m;
  workgroupBarrier();
  for (var off = 128u; off > 0u; off >>= 1u) {
    if (lid.x < off) { red[lid.x] = max(red[lid.x], red[lid.x + off]); }
    workgroupBarrier();
  }
  m = red[0];
  workgroupBarrier();
  var s = 0.0; var sl = 0.0;
  for (var j = lid.x; j < p.V; j += 256u) { let z = logits[o + j] - m; let e = exp(z); s += e; sl += e * z; }
  red[lid.x] = s; red2[lid.x] = sl;
  workgroupBarrier();
  for (var off = 128u; off > 0u; off >>= 1u) {
    if (lid.x < off) { red[lid.x] += red[lid.x + off]; red2[lid.x] += red2[lid.x + off]; }
    workgroupBarrier();
  }
  let tot = red[0];
  let lt = log(tot);
  let t = tgt[gi];
  let c = coef[gi];
  if (lid.x == 0u) {
    stats[gi * 2u] = logits[o + t] - m - lt;
    stats[gi * 2u + 1u] = lt - red2[0] / tot;
  }
  let H = lt - red2[0] / tot;
  workgroupBarrier();
  for (var j = lid.x; j < p.V; j += 256u) {
    let lp = logits[o + j] - m - lt;
    let pj = exp(lp);
    var g = c * pj;
    if (j == t) { g -= c; }
    logits[o + j] = g + p.ent * pj * (lp + H);
  }
}`;

/** Embedding gradient: for each distinct token, sum the rows that used it (no float atomics needed). */
export const EMBED_BWD = /* wgsl */ `
struct P { d: u32, n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> groups: array<u32>; // [token, start, count] per group
@group(0) @binding(2) var<storage, read> rows: array<u32>;
@group(0) @binding(3) var<storage, read> dx: array<f32>;
@group(0) @binding(4) var<storage, read_write> dwte: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let gi = wg.y * 65535u + wg.x;
  if (gi >= p.n) { return; }
  let tok = groups[gi * 3u]; let st = groups[gi * 3u + 1u]; let cnt = groups[gi * 3u + 2u];
  for (var c = lid.x; c < p.d; c += 256u) {
    var s = 0.0;
    for (var j = 0u; j < cnt; j++) { s += dx[rows[st + j] * p.d + c]; }
    dwte[tok * p.d + c] += s;
  }
}`;

/** Sum of squares of a buffer into 256 partials starting at partial[off]. Dispatch exactly 256 workgroups. */
export const SUMSQ = /* wgsl */ `
struct P { n: u32, off: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  var s = 0.0;
  for (var i = g.x; i < p.n; i += 65536u) { let v = x[i]; s += v * v; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var off = 128u; off > 0u; off >>= 1u) {
    if (lid.x < off) { red[lid.x] += red[lid.x + off]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { partial[p.off + wg.x] = red[0]; }
}`;

/** norm = sqrt(sum(partials)); out = [norm, clip scale]. One workgroup. */
export const NORM_FINAL = /* wgsl */ `
struct P { n: u32, maxNorm: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> partial: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var s = 0.0;
  for (var i = lid.x; i < p.n; i += 256u) { s += partial[i]; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var off = 128u; off > 0u; off >>= 1u) {
    if (lid.x < off) { red[lid.x] += red[lid.x + off]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    let norm = sqrt(red[0]);
    out[0] = norm;
    out[1] = min(1.0, p.maxNorm / (norm + 1e-6));
  }
}`;

export const ADAMW = /* wgsl */ `
struct P { n: u32, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, bc1: f32, bc2: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@group(0) @binding(5) var<storage, read> norm: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let scale = norm[1];
  for (var i = gid.x; i < p.n; i += nw.x * 256u) {
    let gi = g[i] * scale;
    let mi = p.b1 * m[i] + (1.0 - p.b1) * gi;
    let vi = p.b2 * v[i] + (1.0 - p.b2) * gi * gi;
    m[i] = mi; v[i] = vi;
    w[i] -= p.lr * ((mi / p.bc1) / (sqrt(vi / p.bc2) + p.eps) + p.wd * w[i]);
  }
}`;
