import Foundation
import CoreGraphics

// Export-only entry point: exercise the existing simulator without model weights.
// This is a functional smoke check, not a benchmark or a model-quality claim.
extension BatchSimConfig {
    static var publicSmoke: BatchSimConfig {
        let args = CommandLine.arguments
        let output: String
        if let i = args.firstIndex(of: "--output"), i + 1 < args.count {
            output = args[i + 1]
        } else {
            output = FileManager.default.temporaryDirectory
                .appendingPathComponent("mac-rc-simulation-smoke").path
        }
        return BatchSimConfig(
            runs: [BatchSimRunSpec(
                runIndex: 1, seed: 6001, trackShape: 0, sceneType: 0,
                cameraPreset: 0, carCount: 5, cameraHeight: 2.5,
                enableRamps: false, recallRate: 0.8236, fpRate: 0.7803,
                forceSimulatedYOLO: true
            )],
            simDuration: 2.0,
            outputDir: output
        )
    }
}
