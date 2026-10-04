# Third-party licences

HydrogenSplat.app compiles in the following open-source code. Each licence's notice is
reproduced here and inside the app (HydrogenSplat ▸ About HydrogenSplat ▸ Acknowledgements;
source: `app/Sources/HSCore/Acknowledgements.swift`). `app/scripts/make_app.sh` also copies
this file into `HydrogenSplat.app/Contents/Resources/`.

| Component | Author | Licence | Pinned | Used for |
|---|---|---|---|---|
| [MetalSplatter](https://github.com/scier/MetalSplatter) (MetalSplatter, SplatIO, PLYIO) | Sean Cier | MIT | `464eb37c55d90d7362a79120fdf8b50d4ae03296`, vendored in `app/Vendor/MetalSplatter` **with one modification** | the in-app splat viewer |
| [spz-swift](https://github.com/scier/spz-swift) | Niantic Labs; Swift port by Sean Cier | MIT | 2.1.0 (`app/Package.resolved`) | linked through SplatIO's .spz reader |

MetalSplatter is modified: one statement of `MetalSplatter/Resources/SplatProcessing.metal`
(the vertex shader no longer converts each splat's colour to linear light), so that the viewer
blends colours the way Brush trains them. What changed and why:
`app/Vendor/MetalSplatter/HYDROGENSPLAT.md`. Its tests, sample apps and `SplatConverter` tool
are not vendored, so `swift-argument-parser` is no longer fetched.

The engine launches Brush, COLMAP / pycolmap, OpenCV, NumPy, FFmpeg and REDline as separate
programs installed by the user. They are not distributed with HydrogenSplat.

## MetalSplatter

```
MIT License

Copyright (c) 2026 Sean Cier

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## spz-swift

```
MIT License

Copyright (c) 2024 Niantic Labs

This is a Swift port of the original C++ implementation by Niantic Labs.
Swift port by Sean Cier, 2026.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

NOTICE:

```
spz-swift
=========

This software is a Swift port of the spz library by Niantic Labs.

Original C++ implementation:
  Repository: https://github.com/nianticlabs/spz
  Copyright: (c) 2024 Niantic Labs
  License: MIT

Swift port by Sean Cier, 2026.
```
