import CoreML
import Foundation
import Testing

@Suite("Selected local FP32 classifier")
@MainActor
struct SelectedModelTests {
    @Test func bundledModelIsTheVerifiedFP32Checkpoint() throws {
        let url = try #require(Bundle.main.url(forResource: "KantoClassifier", withExtension: "mlmodelc"))
        let configuration = MLModelConfiguration()
        #if targetEnvironment(simulator)
        configuration.computeUnits = .cpuOnly
        #else
        configuration.computeUnits = .all
        #endif
        let model = try MLModel(contentsOf: url, configuration: configuration)
        let image = try #require(model.modelDescription.inputDescriptionsByName["image"]?.imageConstraint)
        #expect(image.pixelsWide == image.pixelsHigh && [224, 256].contains(image.pixelsWide))
        let logits = try #require(model.modelDescription.outputDescriptionsByName["logits"]?.multiArrayConstraint)
        #expect(logits.shape.map(\.intValue) == [1, 1025])
        #expect(logits.dataType == .float32)
        let metadata = try #require(model.modelDescription.metadata[.creatorDefinedKey] as? [String: String])
        #expect(metadata["checkpoint_sha256"] == "6d7e865ab59d9298b15c6e4e29aca811398fd040a68a37b25193f04331681e10")
        #expect(metadata["app_labels_sha256"] == "3b794189d9468b353b7508208446d7be428c3841f04d8d47b2a65f898e0d17dc")
    }
}
