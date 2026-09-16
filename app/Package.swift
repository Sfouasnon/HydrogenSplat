// swift-tools-version:5.10
// HydrogenSplat.app — the SwiftUI orchestrator (strategy §2). Open this folder in Xcode, or:
//   swift build && swift run HydrogenSplat        swift test
import PackageDescription

let package = Package(
    name: "HydrogenSplat",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "HydrogenSplat", targets: ["HydrogenSplat"]),
    ],
    targets: [
        // Everything that is not a view: event parsing, the process runner, manifests, projects.
        .target(name: "HSCore", path: "Sources/HSCore"),
        .executableTarget(name: "HydrogenSplat", dependencies: ["HSCore"], path: "Sources/HydrogenSplat",
                         resources: [.copy("Resources")]),
        .testTarget(name: "HSCoreTests", dependencies: ["HSCore"], path: "Tests/HSCoreTests",
                    resources: [.copy("Fixtures")]),
    ]
)
