# Capture SOP v1 — shooting something that will actually solve

Capture decides more than any training flag. Every failure in this repo's history traces back
to the shoot, not the trainer: the 09-13 spec's "what made it work" is a list of capture
choices, cap064 is a bad photograph, and the 2026-09-20 iPhone take lost 14 frames to a
pull-back at the end. You cannot fix any of that later.

This covers three sources: the H1 stereo clip, a mono video (iPhone), and a camera array.
Numbers here are measured from projects in this repo, not guessed. Where something is a
guess, it says so.

---

## 1. The rules that hold for every source

**Translate, don't pan.** Parallax comes from moving the camera through space. A pan gives
large optical flow and near-zero baseline; the essential matrix degenerates and the solve
collapses (observed: 1.7–2.1 mm camera steps, negative scales, 16/68 registered). This is why
`select_frames.py` fits a homography and selects on the *residual after removing it* — rotation
is exactly a homography, so what survives is the depth-bearing translation. Do not add a
"flow > N" escape hatch; it reintroduces the pan problem.

**One continuous move.** rig6, the take everything else is measured against: a low pass
right→left→right, then rise and cross at height, then drop for a second low pass. Azimuth −26°
to +50°, path length 1,729 mm.

**Cover the same azimuths twice, at two heights.** The elevation rings (`engine/hs/bands.py`)
are low [−25°, +5°), mid [+5°, +20°), high [+20°, +45°); the azimuth reporting band is 45°.
The coverage maps keep coming back empty at *high*, so deliberately get above the subject on
one pass. The ring edges are the observed spans of existing captures, not validated optima —
move them once something shot *to* them has been scored.

**Hold a roughly constant distance.** Measured camera-to-subject:

| project | distance (mm) | median |
|---|---|---|
| 2026-09-13_coins | 206–328 | 277 |
| 2026-09-15_head | 494–845 | 698 |

**Stay under ~25°/s.** Above that, frames smear and dwell stops accruing in the capture guide.

**Don't pull back at the end.** The 09-20 iPhone take's last 3.8 s were a pull-away: the
subject shrank, the frame filled with featureless table, and all 13 selected frames from that
stretch failed to register. A camera that sees too little of what the others see cannot be
placed.

**Give the scene texture.** SfM finds features; blank surfaces have none. The 09-20 take was
shot on a white table and ran a **median 50% of each frame below 2 DN of local variance** —
half of every frame carried no features at all, and 58 of 115 picks were over half. A
near-planar subject surrounded by blank surface is the classic degenerate case. A ChArUco board
laid flat beside the subject fixes this *and* gives scale (§4).

**Watch clipping, but measure it correctly.** Count *truly saturated* pixels (all channels
≥254), not "bright". A white subject on a white table reads ~19.5% bright and only 2.87% truly
saturated, and the detail in those bright regions is intact. Coins ran 0.64%. Clipped pixels
carry no gradient and cannot be reconstructed.

---

## 2. Before you press record

- [ ] Subject static. Nothing moves between the first frame and the last.
- [ ] A **scale reference in the scene** (§4). This is the only item that cannot be added later.
- [ ] Texture in the scene — a board, a cloth, anything but blank surface.
- [ ] Lock exposure and focus.
- [ ] **Stabilisation off** (§3.2 — it is not just a crop).
- [ ] One lens. Don't change distance enough to make a phone switch cameras.
- [ ] Plugged in, if the shoot feeds straight into a long run.

---

## 3. Per-source notes

### 3.1 H1 stereo clip

The calibrated baseline gives every solve millimetres for free — the one thing the stereo rig
buys that nothing else does. Profile is matched on the clip's `leia3d_*` tags; ingest refuses
anything that is not one 3840×1080 stream. Video-mode calibration does **not** transfer from
stills calibration.

### 3.2 Mono video (iPhone)

The selector was never stereo — it walks one view, so `select_frames.py --mono` uses the whole
frame instead of cropping the left eye. Verified against the stereo path on a cropped H1 clip:
7 of 8 picks identical, the eighth one frame off from a re-encode tie-break.

**Frame rate.** `--min-gap` / `--max-gap` are frame counts calibrated at 30 fps. At 60 fps
double them (`--min-gap 12 --max-gap 180`) or the floor halves in real time.

**The rotation tag.** iOS records sensor-native landscape and adds a rotate tag. OpenCV
*honours* it, so a portrait-shot clip decodes 1080×1920 and any calibration from it lands in
the wrong axes. Extract with `ffmpeg -noautorotate` to get sensor-native 1920×1080 frames,
which is the orientation a landscape clip already has.

**Stabilisation.** Measured on 2026-09-20, two independent ways:

| | fx | hFOV | 35 mm equiv |
|---|---:|---:|---:|
| ChArUco calibration (1080p30 board clip) | 1672.0 | 59.73° | 31.3 mm |
| COLMAP self-calibration (1080p60 card clip) | 1756.0 | 57.33° | 32.9 mm |
| an uncropped ~26 mm main camera would be | ~1387 | — | 26 mm |

Both clips are cropped ~20% from native, and they disagree with each other by **5.03%** — the
1080p60 take, which had more motion, is cropped harder. Meanwhile `k1` agrees to **0.88%**
(0.19945 vs 0.19769), so both are certainly seeing the same lens. That 5% is the "one shared
camera" assumption failing: with stabilisation on, intrinsics are a property of the *take*, not
of the camera, so a calibration clip cannot calibrate a different take. Turn it off.

**Decoding is the slow part.** HEVC through OpenCV ran ~6.4 fps; the same footage re-encoded to
x264 ran ~36 fps. Transcode before selecting on a long clip.

### 3.3 Camera array

One frame per camera through `hs ingest --frames DIR` (or `--r3d`). Frame names become view
names, so **letters, digits and `-` only** — an underscore collides with the `_L`/`_R` suffix
`rig.py` strips. There is nothing to select: ingest marks `select` done.

A mono *video* can ride this route too — hand it a folder of selected frames — at the cost of
`hs select`'s parallax picking.

---

## 4. Scale

Photogrammetry from unknown cameras has no unit, and scale is a property of each
reconstruction — it does **not** transfer between takes the way calibration does. It is not
cosmetic: the mask radius is in metres, and `--subject-mm`, `--headroom-mm`, move framing and
the coverage rings are all mm.

| source | how it gets metres |
|---|---|
| H1 stereo | free, from the calibrated baseline |
| array | `--scale-pair CAM1,CAM2,MM` from a measured camera spacing |
| mono video | nothing automatic — `scene_scaled` fails and says so |

For a mono take, put a **ChArUco board flat in the scene**. Better than a ruler: the squares
are machine-detectable, so the factor can be computed rather than eyeballed. Place it off to
one side — it only has to be in the reconstruction, not in the camera move you will render, or
you will be masking it out of the splat afterwards.

`hs solve --scale S` can be applied retroactively at any time, so a good take without a
reference is recoverable if you can measure **one** real distance in the scene later. Do that
before the scene is disturbed.

Board note: the board's physical square size does not affect intrinsics at all — only the
translations. So a calibration clip can be shot with any board.

---

## 5. What good looks like

Check these before training on anything.

| stage | number | good | measured |
|---|---|---|---|
| select | frames | 30–120 | coins 70, iPhone 115 |
| select | median gap | 6–15 frames | — |
| select | max-gap picks | < 15% | — |
| solve | registered | **100%** | coins 140/140, iPhone 101/101 after trimming |
| solve | mean reprojection | ≤ 1.8 px | coins 1.30, iPhone **0.717** |
| solve | worst image | no image > 3 px | iPhone 1.63 |
| solve | features/image | thousands | iPhone median 4,507 |
| ingest | truly saturated | low single % | coins 0.64%, iPhone 2.87% |

A partial solve is not trainable. `all_frames_registered` failing means some cameras could not
be placed; find out which, and whether they share a cause, before going further.

---

## 6. Failure signatures

| what you see | what it means |
|---|---|
| a run of unregistered frames at one end | the camera left the subject — a pull-back or a walk-away |
| one isolated unregistered frame | a bad photograph: blur, occlusion, or a jump |
| `frames differ in size: []` | nothing was readable — check for dangling symlinks before suspecting the images |
| solve collapses, tiny camera steps | panning instead of translating |
| `scene_scaled` fails | expected on mono without a reference; see §4 |
| render out-resolves the photograph | motion blur in the capture, not a sharp model |
| self-calibrated focal ≠ your calibration | stabilisation, or a different capture mode |
