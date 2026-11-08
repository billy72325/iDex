import AVFoundation
import CryptoKit
import Foundation
import Testing
@testable import iDex

@MainActor
@Suite("Bundled Pokémon cries", .serialized)
struct PokemonCryTests {
    @Test func everyCatalogEntryHasItsOwnVerifiedPlayableCry() throws {
        struct Source: Decodable {
            let id: Int
            let name: String
            let file: String
            let source: String
            let sha256: String
            let frames: Int
            let sampleRate: Double
        }
        let manifestURL = try #require(Bundle.main.url(forResource: "cry-sources", withExtension: "json"))
        let sources = try JSONDecoder().decode([Source].self, from: Data(contentsOf: manifestURL))
        #expect(sources.count == 1025)
        #expect(Set(sources.map(\.id)) == Set(PokemonCatalog.shared.entries.map(\.id)))
        #expect(Set(sources.map(\.file)).count == 1025)
        for source in sources {
            #expect(source.name == PokemonCatalog.shared.pokemon(id: source.id)?.englishName)
            #expect(source.source.hasSuffix("/latest/\(source.id).ogg"))
            let resource = String(format: "pokemon-cry-%03d", source.id)
            #expect(source.file == "\(resource).wav")
            let url = try #require(Bundle.main.url(forResource: resource, withExtension: "wav"))
            let digest = SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
            #expect(digest == source.sha256)
            let buffer = try AudioPlaybackWorker.cryBuffer(dexID: source.id)
            #expect(buffer.format.sampleRate == source.sampleRate)
            #expect(buffer.frameLength == source.frames)
            let channel = try #require(buffer.floatChannelData?[0])
            #expect((0..<Int(buffer.frameLength)).contains { abs(channel[$0]) > 0.001 })
        }
    }
}
