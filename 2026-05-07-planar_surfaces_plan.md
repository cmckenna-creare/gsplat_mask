# Plan: Plane-associated 2D Gaussian splats

## Design summary

Replace per-splat `means` with `(plane_id, u, v)` and constrain `quats` so each splat's normal matches its plane's normal. World position is *derived* every forward pass:

```
world_xyz[i] = origin[plane_id[i]] + u[i] * basis_u[plane_id[i]] + v[i] * basis_v[plane_id[i]]
```

Loss attribution is geometric residual: how far each splat would *want* to drift from the plane, measured by gradient-on-`means` projected along the plane normal — not a separate forward pass. Surfaces split via 2-means on `(u, v)` when their accumulated residual exceeds threshold. Surfaces merge via spatial+normal hashing, not pairwise.

Scope: 2DGS only (`examples/simple_trainer_2dgs.py`). PPISP is out of scope.

## 1. Data model (new `PlaneState` object)

A single `PlaneState` dataclass owned by the runner, holding all plane-level state. Fields:

| Field                | Shape       | Meaning                                       | Trainable                          |
| -------------------- | ----------- | --------------------------------------------- | ---------------------------------- |
| `origin`             | `[P, 3]`    | plane origin in world space                   | yes                                |
| `normal_raw`         | `[P, 3]`    | unnormalized normal (normalized at use)       | yes                                |
| `basis_u_seed`       | `[P, 3]`    | frozen seed for orthonormal basis             | no                                 |
| `splat_uv`           | `[N, 2]`    | per-splat in-plane coords                     | yes (replaces `means`)             |
| `splat_quat_yaw`     | `[N]`       | in-plane rotation only                        | yes (replaces `quats`)             |
| `splat_plane_id`     | `[N]` int64 | assignment                                    | no (re-assigned at merge/split)    |
| `residual_ema`       | `[P]`       | EMA of attributed loss per plane              | no                                 |
| `splat_residual_ema` | `[N]`       | EMA of per-splat normal-direction grad        | no                                 |

- **Why `normal_raw` not just normal:** lets gradient flow naturally; we normalize at use.
- **Why `basis_u_seed` not the basis itself:** the basis is frozen at creation. We store a seed direction; the actual orthonormal `(basis_u, basis_v)` is computed deterministically as `normalize(seed - (seed·n)n)` and `n × basis_u` each forward. This keeps the basis valid even as `normal_raw` learns.
- **Why `splat_quat_yaw`:** the splat's full quaternion is reconstructed as `align_z_to(plane_normal) ∘ rotate_z(yaw)`, so the normal constraint is structural, not penalty-based.

The `means` and `quats` parameters are **removed** from `self.splats`. Their optimizer entries are removed too. New optimizer entries: `origin`, `normal_raw`, `splat_uv`, `splat_quat_yaw`.

## 2. Forward pass — `gsplat/planes.py` (new module)

Two pure functions, called from `Runner.rasterize_splats` *before* `rasterization_2dgs`:

- `compute_basis(normal_raw, basis_u_seed) -> (n, u, v)` — `[P, 3]` each.
- `assemble_splats(planes, splat_uv, splat_quat_yaw, splat_plane_id) -> (means, quats)` — gathers plane params by id, reconstructs world `means [N, 3]` and `quats [N, 4]`.

Edit `simple_trainer_2dgs.py:549–553` to call `assemble_splats` instead of reading `self.splats["means"/"quats"]`. The rest of the rasterizer call is unchanged. Backprop flows through `assemble_splats` to `origin`, `normal_raw`, `splat_uv`, `splat_quat_yaw` automatically.

## 3. Initialization

In `create_splats_with_optimizers` (`simple_trainer_2dgs.py:280–327`):

- Compute initial `points` and `quats` as today.
- Set `P = N`, `splat_plane_id = arange(N)`.
- `origin = points` (every splat starts as its own plane centered on itself).
- `normal_raw =` third column of rotmat from initial random quat (the 2DGS convention: disk normal = `R[:, 2]`).
- `basis_u_seed =` first column of that rotmat.
- `splat_uv = zeros(N, 2)` (each splat sits at its plane origin).
- `splat_quat_yaw = zeros(N)`.
- Drop `means` and `quats` from the param dict; add the four new ones.

Learning rates (defaults to start):

- `origin: 1.6e-4 * scene_scale` (same as old `means`)
- `normal_raw: 1e-3`
- `splat_uv: 1.6e-4 * scene_scale`
- `splat_quat_yaw: 1e-3`

## 4. Merge step — every `N=500` steps

Goal: O(N) approximate, not O(P²). Uses spatial+normal hashing.

```
for each plane p:
  bin = (
    quantize(normal_p, cosine_bins=32),         # discretize unit sphere
    quantize(origin_p · normal_p, step=tau_d),  # signed distance from world origin
  )
group planes by bin
within each bucket: union-find merge any pair with
  cos(n_p, n_q) > 0.995 AND |Δorigin · n| < 0.02 * scene_scale
```

Merging plane `q` into `p`:

1. New `origin_p ← centroid of all assigned splat world positions` (approximate is fine).
2. New `normal_p ← normalize(weighted_sum(normals, weights=splat_count))`.
3. Re-derive `basis_u_seed_p` from old `basis_u_seed_p` (keep continuity).
4. For each splat that was in `q`, recompute its `(u, v)`: project its current world position onto the new plane → new `(u, v)`. Recompute `splat_quat_yaw` likewise (project old in-plane orientation).
5. `splat_plane_id` of `q`'s splats ← `p`.
6. Mark `q` for removal.

After all unions: compact the plane arrays (remove dead plane slots), update `splat_plane_id` to the compacted indices, update the optimizer state for plane params using the existing `_update_param_with_optimizer` helper from `ops.py`.

**Note on optimizer state:** `splat_uv` and `splat_quat_yaw` values *change* during merge. Adam's running moments for those entries become stale; zero them for affected splats (same pattern as `split` in `ops.py:184–186`).

## 5. Split step — every `N=500` steps (same callback, after merge)

For each plane `p` with `residual_ema[p] > tau_split`:

1. Run 2-means on `splat_uv` of splats assigned to `p` (one iteration of Lloyd's; speed > accuracy).
2. Allocate one new plane `p'`. Initialize `origin_{p, p'}` to the two cluster centroids (lifted to world via the *current* plane), `normal` and `basis_u_seed` copied from `p`.
3. Reassign half of the splats to `p'`, recompute their `(u, v)` relative to the new origin.
4. Reset `residual_ema[p]` and `residual_ema[p']`.

`residual_ema` is updated during `step_post_backward`: project `splat_uv.grad` and `origin`-relative position drift along the normal, EMA per plane (β=0.9). One pass over the grad tensors, no extra rasterization.

## 6. Densification interaction

`DefaultStrategy` calls `duplicate`, `split`, `remove` from `gsplat/strategy/ops.py`. These operate generically over `params`. Two changes needed:

- **Add `splat_plane_id` to `state`** (not `params`) so the existing `for k, v in state.items()` loops in `ops.py:133`, `ops.py:191`, `ops.py:223` carry it along automatically. Same for `splat_residual_ema`.
- **Override `split`** to handle the per-splat `(u, v)` jitter: in `ops.py:139`'s `param_fn`, when `name == "splat_uv"`, jitter in plane-local space (use the splat's plane scale) instead of the world-space rotation/scale jitter currently done for `means`. Cloned splats inherit parent `splat_plane_id` (already free via state-carry).

When `remove` is called, also run a **garbage-collect pass** afterward: any plane with zero assigned splats is removed and indices compacted. This is cheap (`bincount(splat_plane_id) == 0`).

## 7. Hyperparameter defaults

| Knob                                  | Default                       | Notes                                      |
| ------------------------------------- | ----------------------------- | ------------------------------------------ |
| Merge/split interval                  | 500 steps                     | Aligned with typical densification interval |
| Cosine-similarity threshold (merge)   | 0.995                         | ≈ 5.7°                                     |
| Origin distance threshold along normal | `0.02 * scene_scale`         | Scale-invariant                            |
| Spatial hash bins (normal)            | 32                            | Approximate, fast                          |
| Spatial hash bins (signed distance)   | `2 * threshold` width         |                                            |
| Split residual threshold              | `1.5 * median(residual_ema)`  | Adaptive — splits the worst ~25%           |
| Min splats per plane                  | 4                             | Don't split below this                     |
| Max planes split per pass             | 5% of total                   | Cap thrash                                 |
| EMA decay                             | 0.9                           |                                            |

All exposed on the trainer `Config` dataclass (around `simple_trainer_2dgs.py:115`) with sensible defaults so existing runs are unaffected when the feature is disabled.

## 8. Feature flag

Add `cfg.use_planar_surfaces: bool = False` to the `Config`. When `False`, the trainer uses the existing `means`/`quats` path — zero behavior change. When `True`, the parameter dict and forward path swap to the plane-based one. This lets you A/B without forking the trainer.

## 9. Files touched

| File                                | Change                                                                                                                                                                                                                                          |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gsplat/planes.py`                  | **New.** `PlaneState`, `compute_basis`, `assemble_splats`, `merge_planes`, `split_planes`, `compact_planes`, `accumulate_residuals`.                                                                                                          |
| `examples/simple_trainer_2dgs.py`   | Config flag; conditional init in `create_splats_with_optimizers`; conditional `assemble_splats` call in `rasterize_splats`; merge/split callback every N steps in `train()` near the strategy callbacks (line ~981); checkpoint save/load includes `PlaneState`. |
| `gsplat/strategy/ops.py`            | Tiny: in `split`, branch on `splat_uv` for in-plane jitter. (Or override only in a subclass to avoid touching shared code — see risk note below.)                                                                                              |

No CUDA changes. No changes to `rasterization_2dgs` itself — it still receives standard `means`/`quats`.

## 10. Phasing

1. **Data model + forward pass + init**, no merge/split yet. Train with feature flag on; verify visual output matches feature-off baseline (modulo random init differences). This validates the reparam.
2. **Garbage collection + densification interaction.** Run with `DefaultStrategy` on, confirm no crashes after clone/split/prune.
3. **Merge logic.** Every 500 steps. Log `num_planes` to TB.
4. **Split logic.** Add residual EMA + 2-means split.
5. **Tune defaults.** Watch `num_planes` trajectory and final visual quality.

## 11. Risks / open questions

- **Touching `ops.py` is shared with 3DGS.** The `splat_uv` branch in `split` won't fire when 3DGS calls it (no such param), so it's safe — but a cleaner design would be a `Planar2DGSStrategy` subclass that overrides only this. Lean toward the subclass.
- **2DGS quat → normal convention.** Assuming the disk normal is `R[:, 2]` (third column of rotmat from quat). Worth a 5-min check in `gsplat/cuda/_torch_impl_2dgs.py` before implementation — if it's `R[:, 0]`, the `align_z_to` becomes `align_x_to`.
- **Loss is geometric residual only.** Keeps splats *on* their planes but won't directly catch "this surface is a poor fit to the actual scene geometry" — that signal arrives indirectly via per-splat gradients pulling them off the plane. If quality suffers, revisit.
- **PPISP interaction.** Out of scope; feature flag is independent of `cfg.post_processing`.
