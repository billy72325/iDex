import os

/// Inspect these intervals with Instruments' Points of Interest to distinguish
/// capture, inference, and audio startup on a physical device.
nonisolated enum DexPerformance {
    private static let signposter = OSSignposter(subsystem: "billy.iDex", category: "performance")

    static func begin(_ name: StaticString) -> OSSignpostIntervalState {
        signposter.beginInterval(name, id: signposter.makeSignpostID())
    }

    static func end(_ name: StaticString, _ state: OSSignpostIntervalState) {
        signposter.endInterval(name, state)
    }
}
