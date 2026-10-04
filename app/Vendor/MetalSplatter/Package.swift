// swift-tools-version: 6.1
// MetalSplatter 464eb37c55d90d7362a79120fdf8b50d4ae03296 (https://github.com/scier/MetalSplatter, MIT),
// vendored into HydrogenSplat with one changed shader line: see HYDROGENSPLAT.md.
// The three library targets the app links, as upstream declares them. Left out: the tests and
// their data, SampleApp, SampleBoxRenderer and SplatConverter (and with it swift-argument-parser).
// The folder and the package must both stay named "MetalSplatter": the resource bundle is
// MetalSplatter_MetalSplatter.bundle, which the app and app/scripts/make_app.sh look for.

import PackageDescription

let package = Package(
    name: "MetalSplatter",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
        .visionOS(.v2),
    ],
    products: [
        .library(
            name: "PLYIO",
            targets: [ "PLYIO" ]
        ),
        .library(
            name: "SplatIO",
            targets: [ "SplatIO" ]
        ),
        .library(
            name: "MetalSplatter",
            targets: [ "MetalSplatter" ]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/scier/spz-swift.git", exact: "2.1.0"),
    ],
    targets: [
        .target(
            name: "PLYIO",
            path: "PLYIO",
            sources: [ "Sources" ]
        ),
        .target(
            name: "SplatIO",
            dependencies: [
                "PLYIO",
                .product(name: "spz", package: "spz-swift"),
            ],
            path: "SplatIO",
            sources: [ "Sources" ]
        ),
        .target(
            name: "MetalSplatter",
            dependencies: [ "PLYIO", "SplatIO" ],
            path: "MetalSplatter",
            sources: [ "Sources" ],
            resources: [ .process("Resources") ]
        ),
    ],
    swiftLanguageModes: [.v6]
)
