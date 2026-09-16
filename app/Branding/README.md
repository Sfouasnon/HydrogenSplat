# HydrogenSplat brand

| file | use |
|---|---|
| `hs-mark.svg` | the mark on dark grounds |
| `hs-mark-light.svg` | the mark on light grounds (deeper view colours) |
| `hs-mark-mono.svg` | one colour (`currentColor`), for stamps, favicons, engraving |
| `hs-logo-dark.svg` / `hs-logo-light.svg` | horizontal lockup, text outlined (no font needed) |
| `hs-appicon.svg` | app icon, 64 px and up |
| `hs-appicon-small.svg` | app icon at 16–32 px (no lens detail) |
| `AppIcon.iconset/` | macOS icon set; build the .icns with `iconutil` |
| `AppIcon-1024.png` | 1024 px master |
| `tokens.json` | colours (dark / light), type, radius, mark rules |
| `build.py`, `iconset.py` | regenerate everything (fontTools + Playwright; needs `Michroma.ttf`, OFL, from google/fonts) |

**The mark.** An H of four slats — the four views of a stereo capture, offset vertically like
parallax slices of a hologram — crossed by a single red Gaussian splat that doubles as a
camera's record tally. Cyan → violet runs left to right across the views, like the diffraction
on a holographic foil.

**Rules.** Keep one slat width clear around the mark. Don't recolour the slats individually,
rotate the mark, or put the colour mark on a mid-grey or saturated ground (use mono there).
Red is the tally: one red thing per screen — the splat, a record/progress state, or the SPLAT
half of the wordmark — never decoration.

**Originality.** The identity is original. It borrows the *vernacular* of cinema cameras
(machined black bodies, heat-sink grooves, a lens mount, a red tally light) and of lenticular
4V holography, not any manufacturer's logo, wordmark or product styling.
