# Mask review: the contract between the engine (`hs masks`, `hs train`) and the app's Masks panel

Version 1, 2026-10-04. The engine is the source of truth (`engine/hs/maskcheck.py` measures,
`engine/hs/maskreview.py` keeps the report and the decisions). The app only reads `review.json`,
shows the pictures the engine wrote, and runs `hs masks --decide`.

## Why
A subject mask that leaves out part of the subject teaches the trainer that this part is empty
from that direction (`--alpha-mode transparent`), and the model paints it black from there. On
2026-09-28_Stormtrooper_iPhone 36 of 243 masks did this and were found in four rounds, three of
them after a 2 h 40 min train each. The engine now checks the masks against each other right
after building them, and `hs train` refuses to start while a flagged view has no decision.

## What the check does
Two tests, from the masks and the poses alone (no trained model):

- **A — does the mask hold the subject?** A sparse SfM point that lands inside the mask in ≥ 90 %
  of its views (and is in frame in ≥ 30 % of all views) is a point of the subject. A mask that
  holds under 90 % of those, of the ones in its frame, is flagged (`holds_subject`).
- **B — does the mask agree with the others?** The masks that pass A vote a visual hull on a grid
  of 128 voxels along the longest side: a voxel is solid for view *i* when ≥ 97 % of the *other*
  views with it in frame have it inside their mask. That hull is projected into view *i* at
  480 px on the long edge. What lies outside mask *i* is split into a thin rim and a compact
  piece by a morphological opening (2.3 % of the long edge, 11 px). A piece of ≥ 0.3 % of the
  hull's area in that view is flagged (`piece_outside`).
- A view where Vision found no object it could accept, so the rough region was used, is flagged
  whatever it looks like (`fell_back`), and has no vote.

Measured on the helmet (243 views, 24 s on two cores): 45 flagged, 19 by A and 26 by B, with 34
of the 36 known bad masks among them. The two it misses leave out 0.2–0.3 % of the helmet. Of
the 11 flagged that were not known, the ones looked at leave out the dark opening under the
helmet, or are air in front of the face.

### Limits
- The thresholds were chosen on one project, knowing the answer.
- The voted hull is too large where few views are tangent to the surface, so B flags masks that
  are right (at least two of 45 on the helmet: air in front of the face). **The check does not decide; it
  chooses what a person looks at.**
- A bite shared by more than 3 % of the voting views carves the hull itself and is not seen.
  With fewer than 34 voting views the vote is unanimous: one bad mask carves the hull for every
  other view (nothing is flagged wrongly, but a second mask with the same bite hides behind it).
- Under 8 masks that pass A: only A runs (`note` says so). Under 50 subject points: nothing is
  judged except fall-backs (`note` says so).
- Only masks that are too small are tested. A mask that takes in table or reflection is not.

## Repairs
For a flagged view the engine prepares, when it can, a mask to use instead:

| kind | what it is | approximate |
|---|---|---|
| `reselect` | Vision's own object for this view (still in `masks_vision/`), which the build had turned down because it lay outside the rough region; kept when ≥ 85 % of it lies inside the hull | no |
| `reselect` | the same, plus a missing piece taken from the hull | yes |
| `fill` | a hole inside the mask, closed | no |
| `fill` | a missing piece bounded by the mask and the frame edge, filled up to the frame edge | yes |
| `hull` | a missing piece on the silhouette, added with the hull's own outline (about 1 px of the 480 px check image on average, 4 px at worst: 8 and 30 px of a 3840 px frame) | yes |
| `hull` | the whole mask drawn from the hull, for a fall-back with no object from Vision | yes |

`approximate: false` means the repair needs no look: it is a real segmentation that the hull
confirms, or a hole. `approximate: true` means the repair rests on the hull being right about
where the subject is, and the hull is too large exactly where a bite would be. On the helmet 5
of 45 repairs are not approximate and 40 are; the 40 include the real bites out of the chin and
the dome, and the two pieces of air.

A repair is offered only if the repaired mask passes the same check, puts no more than 3 % of
the subject's area outside the hull, and holds ≥ 90 % of the subject's points. It is finished
the way the build finished its masks (`--feather-px`, `--exclude-highlights`, and `--grow-px`
for an object fresh from Vision).

## Files (all under the project folder)
```
masks_review/review.json                 the report and the decisions
masks_review/flagged_sheet.jpg           the eight worst on one picture
masks_review/<eye>/<cap>.jpg             preview of a flagged view (always present for a flagged view)
masks_review/<eye>/<cap>_repair.jpg      preview of the repaired mask (only when a repair exists)
masks_review/<eye>/<cap>_repaired.png    the repaired mask itself (only when a repair exists)
masks_review/original/<eye>/<cap>.png    the mask as built, kept while a repair is installed
```
`<eye>/<cap>` is the view key, e.g. `L/sel192-01823` (the dataset's `masks/L/sel192-01823.png`).
Paths inside review.json are relative to the project folder. Pictures are 540 px high: the whole
frame, and beside it the piece in question enlarged. Red is this view's mask, blue the subject as
the other views see it, yellow the piece in question; in a repair picture green is the repaired
mask. A check removes the files of views that are no longer flagged.

## review.json
```json
{
  "version": 1,
  "created": "2026-10-04T14:02:11-0700",
  "views": 243,
  "masks_digest": "5c6fb2c8d69aa4681d74ee924909b0e8",
  "method": {"agree_min": 0.9, "hull_threshold": 0.97, "open_frac": 0.023, "min_piece": 0.003,
             "check_edge": 480, "voxel_mm": 2.885, "voters": 224, "subject_points": 15946, "seconds": 23.6},
  "note": null,
  "flagged": [
    {
      "view": "L/sel200-01880",
      "reasons": ["fell_back", "holds_subject"],
      "score": 0.9413,
      "why": "Vision found no object it could accept here, so the rough region was used; it leaves out 94.1% of the subject",
      "agreement": 0.2254,
      "piece_share": 0.9413,
      "fell_back": "nothing inside the geometric mask",
      "preview": "masks_review/L/sel200-01880.jpg",
      "repair": {"kind": "reselect", "approximate": false,
                 "label": "Vision's own object, which the build had turned down",
                 "preview": "masks_review/L/sel200-01880_repair.jpg",
                 "mask": "masks_review/L/sel200-01880_repaired.png"},
      "mask_md5": "0f3c…",
      "decision": null,
      "decided": null
    }
  ],
  "summary": {"flagged": 45, "undecided": 45, "repairable": 45, "exact": 5, "repair": 0, "exclude": 0, "keep": 0}
}
```
- `flagged` is sorted worst first; views whose repair is already installed come last.
- `reasons`: any of `"fell_back"`, `"holds_subject"`, `"piece_outside"`.
- `score`: 0–1, larger is worse; for ordering only.
- `why`: one English clause the app shows as is.
- `agreement`: share of the subject's points inside this mask, or null when not measured.
- `piece_share`: share of the subject's area in this view that the mask leaves out, or null.
- `fell_back`: null, or the reason from the build.
- `repair`: null when the engine has none to offer. `label` is one English phrase for the UI.
- `decision`: null (undecided), `"repair"`, `"exclude"` or `"keep"`. `decided`: ISO time or null.
- `mask_md5`: the mask file the entry is about. **A decision belongs to that file**, not to the
  view name.
- `note`: null, or a sentence when the check could not run fully.
- `summary.repairable`: flagged views with a repair; `summary.exact`: of those, not approximate.
- No NaN. Unknown keys must be ignored; missing optional keys read as null. For `version` > 1
  the app shows "written by a newer engine" and no list.

## Commands
All through the normal `hs` runner: they take the project lock, emit events, exit 0 on success.

Build (as before; now ends with the check, `--no-check` leaves it out):
```
hs masks -p <project> --method vision …
```
A build removes `masks_review/` before it writes the masks. Decisions `exclude` and `keep` carry
over to a mask that comes out byte for byte the same.

Check only, on the masks already on disk (builds nothing, marks nothing stale):
```
hs masks -p <project> --check-only
```
Decisions stay with every mask whose file has not changed; an installed repair keeps its entry.
On masks from a build before this change, the fall-backs are taken from the build's own record
(it names the first twelve); the rest are found by test A.

Decide (fast, no recomputation):
```
hs masks -p <project> --decide L/sel200-01880=repair
hs masks -p <project> --decide L/sel230-02123=exclude,L/sel079-00632=keep
hs masks -p <project> --decide @exact=repair,@undecided=exclude
hs masks -p <project> --decide L/sel200-01880=undo
```
- choice is `repair`, `exclude`, `keep` or `undo`.
- lists: `@undecided`; `@repairable` (undecided, has a repair); `@exact` (undecided, its repair
  is not approximate); `@decided`. `repair` on a list takes what can be repaired and leaves the
  rest as it is.
- Tokens are applied one at a time in the order given, so `@exact=repair,@undecided=exclude`
  does not exclude what it has just repaired.
- A command with an unknown view, list or choice, or `repair` for a view without one, is refused
  as a whole (exit 1, an `error` event with a hint) and changes nothing.
- `repair` copies the repaired mask over the dataset mask, keeps the original in
  `masks_review/original/`, and marks train / prune / render / views stale, like a rebuild.
  `undo` of a repair puts the original back. `exclude` and `keep` change nothing in the dataset.
- After every decide the engine rewrites review.json (decisions, `summary`, `masks_digest`) and
  the masks stage's `mask_review` metric and `masks_reviewed` check in manifest.json. The stage's
  own status, argv and other metrics stay as the build left them.

## Manifest (stages.masks)
- metric `mask_review`: the summary plus `"report": "masks_review/review.json"`.
- check `masks_reviewed`: ok when `undecided` is 0; `needs_human` while it is not.

## Train
`hs train` with masks in use refuses to start when
- review.json is missing: hint `hs masks -p <project> --check-only`;
- the masks on disk are not the ones that were reviewed (`masks_digest` differs);
- a flagged view is undecided and not already in `--exclude`.

`--allow-unreviewed-masks` overrides all three. Views decided `exclude` are left out of the run
as if named in `--exclude`. The train stage records `mask_review` (what it found, and
`excluded_by_review`).

## What the panel shows
Inside the Masks group box, under "What was built", whenever mask files exist:
- no review.json: one line and **Check masks** (`--check-only`);
- a header: "Review — N flagged of M views, K undecided", or "All M masks agree", or the note;
- **Apply the N repairs that need no look** (`@exact=repair`), **Exclude the rest**
  (`@undecided=exclude`), **Check again** (`--check-only`);
- one row per flagged view, worst first: the preview, the repair preview beside it, the view
  key, `why`, the repair's `label` (with "approximate, look at the picture first"), and
  **Repair** / **Exclude** / **Keep** — or, once decided, the decision and **Undo**. Clicking a
  picture opens it.

The Train panel shows a warning under Layer when review.json is missing or a flagged view the
run would train on is undecided. It does not recompute `masks_digest`: masks changed by hand
after the check are caught by `hs train` itself.
