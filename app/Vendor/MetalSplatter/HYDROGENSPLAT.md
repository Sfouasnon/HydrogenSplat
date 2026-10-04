# MetalSplatter, as vendored in HydrogenSplat

Upstream: https://github.com/scier/MetalSplatter at `464eb37c55d90d7362a79120fdf8b50d4ae03296`
(MIT, `LICENSE` beside this file). Copied 2026-10-04: `MetalSplatter/{Sources,Resources}`,
`SplatIO/Sources`, `PLYIO/Sources`, byte for byte except the one statement below and the
comments beside it. Not copied: tests
and test data, SampleApp, SampleBoxRenderer, SplatConverter. `Package.swift` is upstream's with
those targets removed and spz-swift pinned to exactly 2.1.0 (the version the app had resolved).

## The change
`MetalSplatter/Resources/SplatProcessing.metal`, the last statement of `splatVertex`:

```
- out.color = half4(sRGBToLinear(srgbColor), splat.color.a);
+ out.color = half4(srgbColor, splat.color.a);
```

## Why
Brush (and the reference 3DGS rasteriser) composites the splats' SH colours as they are,
`pixel = sum(w_i * c_i)`, and that sum is what the loss compares with the photograph's code
values. Upstream raises each colour to 2.2 before blending and renders to an sRGB target, which
encodes on write: `pixel = encode(sum(w_i * c_i^2.2))`. The two agree only where a single splat
owns the pixel. Where several mix they do not, and a colour above 1 gains most: 2.0 becomes 4.6.

Seen on 2026-09-28_Stormtrooper_iPhone, model E, 2026-10-04: white bars across the last slats
of the rear-left vent in the viewer, absent from `brush-path-render`'s frames of the same model.
A numpy renderer matched to Brush to 3-4 / 255 reproduced the bars slat for slat once its blend
was switched to linear light, and showed none with Brush's blend
(project doc `claude/run-e-bad-masks-excluded-2026-10-04.md`).

## What the app does with it
`SplatScene.swift` gives the renderer a `bgra8Unorm` target (not `bgra8Unorm_srgb`), so the
blended value is stored as the code value it is. The multi-stage pipeline (in use whenever a
depth format is set and `highQualityDepth` is true, its default) accumulates in half-float tile
memory, so over-range sums are carried exactly as in Brush until the final write clamps them.

## Updating to a newer upstream
Copy the same four folders over these, re-apply the one line, and check `Package.swift` against
upstream's targets and dependencies.
