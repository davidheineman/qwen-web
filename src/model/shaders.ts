// WGSL compute kernels for the Qwen3 forward pass.
// Weights stay in their original bf16 form (2 per u32) and are widened to f32 in-shader;
// all activations are f32.

const BF16 = /* wgsl */ `
fn bf16lo(u: u32) -> f32 { return bitcast<f32>(u << 16u); }
fn bf16hi(u: u32) -> f32 { return bitcast<f32>(u & 0xffff0000u); }
`;

// Y[M, ldy] (columns yoff..yoff+N) = X[M, K] · W[N, K]^T     (nn.Linear layout, no bias)
// Tile: 16 rows of X × 64 rows of W per workgroup, K stepped 32 at a time.
export const MATMUL = /* wgsl */ `
${BF16}
struct P { M: u32, N: u32, K: u32, ldy: u32, yoff: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

const TM = 16u;
const TN = 64u;
const TK = 32u;
const WS = 33u; // padded stride to avoid bank conflicts
var<workgroup> xs: array<f32, 512>;   // TM * TK
var<workgroup> ws: array<f32, 2112>;  // TN * WS

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let tx = lid.x;
  let ty = lid.y;
  let li = ty * 16u + tx;
  let row0 = wg.y * TM;
  let col0 = wg.x * TN;
  var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  for (var k0 = 0u; k0 < p.K; k0 += TK) {
    for (var i = 0u; i < 2u; i++) {
      let e = li + i * 256u;
      let r = e / TK;
      let c = e % TK;
      let gr = row0 + r;
      var v = 0.0;
      if (gr < p.M) { v = X[gr * p.K + k0 + c]; }
      xs[e] = v;
    }
    for (var i = 0u; i < 4u; i++) {
      let e = li + i * 256u;
      let r = e / 16u;
      let c2 = e % 16u;
      let gn = col0 + r;
      var u = 0u;
      if (gn < p.N) { u = W[(gn * p.K + k0) / 2u + c2]; }
      ws[r * WS + c2 * 2u] = bf16lo(u);
      ws[r * WS + c2 * 2u + 1u] = bf16hi(u);
    }
    workgroupBarrier();
    for (var k = 0u; k < TK; k++) {
      let xv = xs[ty * TK + k];
      acc[0] += xv * ws[tx * WS + k];
      acc[1] += xv * ws[(tx + 16u) * WS + k];
      acc[2] += xv * ws[(tx + 32u) * WS + k];
      acc[3] += xv * ws[(tx + 48u) * WS + k];
    }
    workgroupBarrier();
  }
  let gr = row0 + ty;
  if (gr < p.M) {
    for (var j = 0u; j < 4u; j++) {
      let gn = col0 + tx + 16u * j;
      if (gn < p.N) { Y[gr * p.ldy + p.yoff + gn] = acc[j]; }
    }
  }
}
`;

// Y[r, :] = X[r, :] * rsqrt(mean(X[r,:]^2) + eps) * G     one workgroup per row
export const RMSNORM = /* wgsl */ `
struct P { M: u32, D: u32, eps: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let row = wg.x + wg.y * 65535u;
  if (row >= p.M) { return; }
  let base = row * p.D;
  var s = 0.0;
  for (var d = lid.x; d < p.D; d += 256u) { let v = X[base + d]; s += v * v; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var w = 128u; w > 0u; w >>= 1u) {
    if (lid.x < w) { red[lid.x] += red[lid.x + w]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(red[0] / f32(p.D) + p.eps);
  for (var d = lid.x; d < p.D; d += 256u) { Y[base + d] = X[base + d] * inv * G[d]; }
}
`;

// In-place rotary embedding (rotate_half convention). X: [T, nh, 128]; CS: [T, 64] cos then [T, 64] sin
export const ROPE = /* wgsl */ `
struct P { T: u32, NH: u32, csOff: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<storage, read> CS: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= p.T * p.NH * 64u) { return; }
  let i = idx % 64u;
  let h = (idx / 64u) % p.NH;
  let t = idx / (64u * p.NH);
  let c = CS[t * 64u + i];
  let s = CS[p.csOff + t * 64u + i];
  let base = (t * p.NH + h) * 128u;
  let x1 = X[base + i];
  let x2 = X[base + i + 64u];
  X[base + i] = x1 * c - x2 * s;
  X[base + i + 64u] = x2 * c + x1 * s;
}
`;

// S[off + h*T*T + i*T + j] = q_i,h · k_j,h/2 / sqrt(128)     (causal: j <= i only)
export const SCORES = /* wgsl */ `
struct P { T: u32, H: u32, off: u32, scale: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read_write> S: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let TT = p.T * p.T;
  if (idx >= p.H * TT) { return; }
  let j = idx % p.T;
  let i = (idx / p.T) % p.T;
  let h = idx / TT;
  if (j > i) { S[p.off + idx] = 0.0; return; }
  let kvh = h / 2u;
  let qb = (i * p.H + h) * 128u;
  let kb = (j * (p.H / 2u) + kvh) * 128u;
  var s = 0.0;
  for (var d = 0u; d < 128u; d++) { s += Q[qb + d] * K[kb + d]; }
  S[p.off + idx] = s * p.scale;
}
`;

// Row-wise causal softmax over S, one thread per (h, i) row.
export const SOFTMAX = /* wgsl */ `
struct P { T: u32, H: u32, off: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> S: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let r = gid.x;
  if (r >= p.H * p.T) { return; }
  let i = r % p.T;
  let base = p.off + r * p.T;
  var m = -3.4e38;
  for (var j = 0u; j <= i; j++) { m = max(m, S[base + j]); }
  var sum = 0.0;
  for (var j = 0u; j <= i; j++) { let e = exp(S[base + j] - m); S[base + j] = e; sum += e; }
  let inv = 1.0 / sum;
  for (var j = 0u; j <= i; j++) { S[base + j] *= inv; }
}
`;

// O[i, h, d] = Σ_j P[h, i, j] · V[j, h/2, d]
export const PV = /* wgsl */ `
struct P { T: u32, H: u32, off: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let h = wg.x;
  let i = wg.y;
  let d = lid.x;
  let kvh = h / 2u;
  let base = p.off + (h * p.T + i) * p.T;
  var acc = 0.0;
  for (var j = 0u; j <= i; j++) { acc += S[base + j] * V[(j * (p.H / 2u) + kvh) * 128u + d]; }
  O[(i * p.H + h) * 128u + d] = acc;
}
`;

// A = silu(G) * U
export const SWIGLU = /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read> U: array<f32>;
@group(0) @binding(3) var<storage, read_write> A: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let g = G[i];
  A[i] = g / (1.0 + exp(-g)) * U[i];
}
`;

// X += Y
export const ADD = /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<storage, read> Y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.n) { return; }
  X[i] = X[i] + Y[i];
}
`;

// Embedding lookup for tokens whose id falls in this vocab chunk [r0, r1).
export const EMBED = /* wgsl */ `
${BF16}
struct P { T: u32, D: u32, r0: u32, r1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> TOK: array<u32>;
@group(0) @binding(2) var<storage, read> E: array<u32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;  // over T * D/2
  let half = p.D / 2u;
  if (idx >= p.T * half) { return; }
  let t = idx / half;
  let c2 = idx % half;
  let id = TOK[t];
  if (id < p.r0 || id >= p.r1) { return; }
  let u = E[(id - p.r0) * half + c2];
  X[t * p.D + c2 * 2u] = bf16lo(u);
  X[t * p.D + c2 * 2u + 1u] = bf16hi(u);
}
`;
