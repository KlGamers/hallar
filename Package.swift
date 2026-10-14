// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Hallar",
    platforms: [.iOS(.v15), .macOS(.v12)],
    products: [.library(name: "Hallar", targets: ["Hallar"])],
    targets: [
        .target(
            name: "CHallar",
            path: "core",
            exclude: ["tests", "third_party/vecdex/LICENSE", "third_party/vecdex/VERSION"],
            sources: ["src", "third_party/vecdex/src"],
            publicHeadersPath: "swift-include",
            cxxSettings: [
                .headerSearchPath("include"),
                .headerSearchPath("src"),
                .headerSearchPath("third_party/vecdex/include"),
            ]
        ),
        .target(name: "Hallar", dependencies: ["CHallar"], path: "swift/Hallar"),
        .testTarget(name: "HallarTests", dependencies: ["Hallar"], path: "swift/Tests/HallarTests"),
    ],
    cxxLanguageStandard: .cxx17
)
