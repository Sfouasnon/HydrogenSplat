import Foundation

/// Third-party code that is compiled into HydrogenSplat.app, with the notices its licences
/// require us to ship. The texts live in the binary, not in a resource file: the MIT licence's
/// one condition is that the copyright and permission notices accompany every copy, and a
/// string constant cannot be left out of a bundle by a packaging script.
///
/// Verbatim from the pinned revisions (app/Package.swift); THIRD_PARTY_LICENSES.md at the
/// repository root carries the same texts for anyone reading the source. When a dependency is
/// added or its pin moves, update both — `ViewerTests.testAcknowledgementsCarryTheRequiredNotices`
/// fails if a notice goes missing.
public struct Acknowledgement: Identifiable, Sendable {
    public var id: String { name }
    public let name: String
    public let author: String
    public let license: String
    public let url: String
    public let revision: String
    /// What HydrogenSplat uses it for, in one line.
    public let use: String
    public let licenseText: String
}

public enum Acknowledgements {
    /// Linked into the app binary.
    public static let bundled: [Acknowledgement] = [
        Acknowledgement(
            name: "MetalSplatter", author: "Sean Cier", license: "MIT",
            url: "https://github.com/scier/MetalSplatter",
            revision: "464eb37c55d90d7362a79120fdf8b50d4ae03296",
            use: "Renders Gaussian splats with Metal and reads .ply models (MetalSplatter, SplatIO, PLYIO) — the in-app model viewer.",
            licenseText: metalSplatterLicense),
        Acknowledgement(
            name: "spz-swift", author: "Niantic Labs; Swift port by Sean Cier", license: "MIT",
            url: "https://github.com/scier/spz-swift",
            revision: "2.1.0 or later, as resolved in app/Package.resolved",
            use: "Linked through MetalSplatter's SplatIO (the .spz reader). A Swift port of https://github.com/nianticlabs/spz.",
            licenseText: spzSwiftLicense),
    ]

    /// Tools the engine launches as separate programs. They are installed by the user, not
    /// distributed with the app, so their licences place no notice requirement on it; they
    /// are named because the app would do nothing without them.
    public static let drives: [(name: String, url: String)] = [
        ("Brush", "https://github.com/ArthurBrussee/brush"),
        ("COLMAP / pycolmap", "https://github.com/colmap/colmap"),
        ("OpenCV", "https://opencv.org"),
        ("NumPy", "https://numpy.org"),
        ("FFmpeg", "https://ffmpeg.org"),
        ("REDline (REDCINE-X PRO)", "https://www.red.com"),
    ]

    static let metalSplatterLicense = """
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
"""

    static let spzSwiftLicense = """
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

--- NOTICE ---

spz-swift
=========

This software is a Swift port of the spz library by Niantic Labs.

Original C++ implementation:
  Repository: https://github.com/nianticlabs/spz
  Copyright: (c) 2024 Niantic Labs
  License: MIT

Swift port by Sean Cier, 2026.
"""
}
