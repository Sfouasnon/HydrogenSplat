// swift-tools-version:5.10
// HydrogenSplat.app — the SwiftUI orchestrator (strategy §2). Open this folder in Xcode, or:
//   swift build && swift run HydrogenSplat        swift test
import PackageDescription

let package = Package(
    name: "HydrogenSplat",
    // macOS 15: MetalSplatter's floor (it uses Synchronization.Mutex). Written as a string because
    // `.v15` needs tools-version 6.0, which would also switch this package to the Swift 6 language mode.
    platforms: [.macOS("15.0")],
    products: [
        .executable(name: "HydrogenSplat", targets: ["HydrogenSplat"]),
    ],
    dependencies: [
        // The splat viewer's renderer and .ply reader (MIT — notices in HSCore/Acknowledgements.swift and
        // THIRD_PARTY_LICENSES.md): MetalSplatter 464eb37c (1.0.1 plus what landed after it — the PLY
        // reader accepts SH degree 1 and 2 files, and SplatChunk no longer writes past a splat's SH
        // slot when counts disagree), vendored with one shader line changed so the viewer blends
        // colours the way Brush trains them (Vendor/MetalSplatter/HYDROGENSPLAT.md).
        // Needs a Swift 6.1+ toolchain (Xcode 16.3 or later).
        .package(path: "Vendor/MetalSplatter"),
    ],
    targets: [
        // Everything that is not a view: event parsing, the process runner, manifests, projects.
        .target(name: "HSCore", path: "Sources/HSCore"),
        .executableTarget(name: "HydrogenSplat",
                          dependencies: ["HSCore",
                                         .product(name: "MetalSplatter", package: "MetalSplatter"),
                                         .product(name: "SplatIO", package: "MetalSplatter")],
                          path: "Sources/HydrogenSplat",
                         resources: [.copy("Resources")]),
        .testTarget(name: "HSCoreTests", dependencies: ["HSCore"], path: "Tests/HSCoreTests",
                    resources: [.copy("Fixtures")]),
    ]
)
