import AVFoundation
import Foundation
import os

nonisolated protocol DeviceAudioPlayback: Sendable {
    /// Returns only after the output engine has rendered its first buffer.
    func activate() async -> Bool
    func deactivate() async
    func play(_ cue: DeviceSound, request: AudioPlaybackRequest)
    func playCry(dexID: Int, request: AudioPlaybackRequest, completion: @escaping @Sendable () -> Void)
    func stopSound()
    func speak(_ text: String, languageCode: String, request: AudioPlaybackRequest)
    func stopSpeaking()
}

/// Invalidated immediately by mute, suspend, or a newer request, even while a
/// native audio call occupies the serial queue. The UI never waits for it.
nonisolated final class AudioPlaybackRequest: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private let speechActivity: (@Sendable (Bool) -> Void)?

    init(speechActivity: (@Sendable (Bool) -> Void)? = nil) {
        self.speechActivity = speechActivity
    }

    var isCancelled: Bool { lock.withLock { cancelled } }
    func cancel() { lock.withLock { cancelled = true } }

    func reportSpeechActivity(_ speaking: Bool) {
        guard !isCancelled else { return }
        speechActivity?(speaking)
    }
}

/// Each delegate belongs to one utterance, so a replaced utterance's delayed
/// cancellation cannot extinguish the next reading's light.
nonisolated final class SpeechPlaybackObserver: NSObject, AVSpeechSynthesizerDelegate {
    private let utteranceID: ObjectIdentifier
    private let request: AudioPlaybackRequest

    init(utterance: AVSpeechUtterance, request: AudioPlaybackRequest) {
        utteranceID = ObjectIdentifier(utterance)
        self.request = request
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        if ObjectIdentifier(utterance) == utteranceID { request.reportSpeechActivity(true) }
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        if ObjectIdentifier(utterance) == utteranceID { request.reportSpeechActivity(false) }
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        if ObjectIdentifier(utterance) == utteranceID { request.reportSpeechActivity(false) }
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didPause utterance: AVSpeechUtterance) {
        if ObjectIdentifier(utterance) == utteranceID { request.reportSpeechActivity(false) }
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didContinue utterance: AVSpeechUtterance) {
        if ObjectIdentifier(utterance) == utteranceID { request.reportSpeechActivity(true) }
    }
}

/// One live output engine serves every chime and bundled Pokémon cry. A foreground warmup starts the
/// actual hardware, so the first short cue doesn't depend on speech priming it.
/// Native setup and speech calls stay off the UI and cooperative executors.
nonisolated final class AudioPlaybackWorker: DeviceAudioPlayback, @unchecked Sendable {
    private static let logger = Logger(subsystem: "billy.iDex", category: "audio")
    private let queue = DispatchQueue(label: "app.idex.audio", qos: .userInitiated)
    private let soundVolume: Float
    private let speechVolume: Float
    private let rendered = AudioRenderProgress()
    private var engine: AVAudioEngine?
    private var player: AVAudioPlayerNode?
    private var buffers: [DeviceSound: AVAudioPCMBuffer] = [:]
    private var currentRequest: AudioPlaybackRequest?
    private var currentCue: DeviceSound?
    private var synthesizer: AVSpeechSynthesizer?
    private var speechObserver: SpeechPlaybackObserver?
    private var voices: [String: AVSpeechSynthesisVoice] = [:]

    init(soundVolume: Float = 0.32, speechVolume: Float = 0.9) {
        self.soundVolume = soundVolume
        self.speechVolume = speechVolume
    }

    func activate() async -> Bool {
        await withCheckedContinuation { continuation in
            queue.async {
                let session = AVAudioSession.sharedInstance()
                do {
                    try session.setCategory(.playback, mode: .default, options: [.mixWithOthers])
                } catch {
                    Self.logger.error("Audio configuration failed: \(error.localizedDescription, privacy: .public)")
                    continuation.resume(returning: false)
                    return
                }
                session.activate(options: []) { active, error in
                    self.queue.async {
                        guard active, error == nil else {
                            Self.logger.error("Audio activation failed: \(error?.localizedDescription ?? "not activated", privacy: .public)")
                            continuation.resume(returning: false)
                            return
                        }
                        do {
                            try self.startEngine()
                            self.waitForOutput(until: .now.advanced(by: .seconds(2)), completion: continuation)
                        } catch {
                            Self.logger.error("Audio engine failed: \(error.localizedDescription, privacy: .public)")
                            self.failActivation(continuation)
                        }
                    }
                }
            }
        }
    }

    private func startEngine() throws {
        if engine == nil {
            guard let format = AVAudioFormat(standardFormatWithSampleRate: DeviceSoundWave.sampleRate, channels: 1) else {
                throw PlaybackError.invalidFormat
            }
            let engine = AVAudioEngine()
            let player = AVAudioPlayerNode()
            let chimes = AVAudioMixerNode()
            engine.attach(player)
            engine.attach(chimes)
            try engine.connectNode(player, to: chimes, format: format)
            try engine.connectNode(chimes, to: engine.mainMixerNode, format: format)
            // Observe real rendered PCM before the final gain. Tests mute that
            // gain while still proving a cue reached the live render graph.
            let progress = rendered
            try chimes.installAudioTap(onBus: 0, bufferSize: 256, format: format) { buffer, _ in
                progress.record(buffer)
            }
            engine.mainMixerNode.outputVolume = soundVolume
            for cue in DeviceSound.allCases {
                let samples = DeviceSoundWave.samples(for: cue)
                guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
                      let channel = buffer.floatChannelData?[0] else { throw PlaybackError.invalidFormat }
                buffer.frameLength = AVAudioFrameCount(samples.count)
                samples.withUnsafeBufferPointer { source in
                    if let base = source.baseAddress { channel.update(from: base, count: source.count) }
                }
                buffers[cue] = buffer
            }
            self.engine = engine
            self.player = player
        }
        guard let engine, let player else { throw PlaybackError.invalidFormat }
        if !engine.isRunning {
            rendered.reset()
            engine.prepare()
            try engine.start()
            // No tone is scheduled during startup: output begins with silence.
            try player.playAudio()
        }
    }

    private func waitForOutput(until deadline: ContinuousClock.Instant, completion: CheckedContinuation<Bool, Never>) {
        if rendered.snapshot.frames > 0 {
            completion.resume(returning: true)
        } else if ContinuousClock.now >= deadline {
            Self.logger.error("Audio output did not begin rendering")
            failActivation(completion)
        } else {
            queue.asyncAfter(deadline: .now() + .milliseconds(10)) {
                self.waitForOutput(until: deadline, completion: completion)
            }
        }
    }

    private func failActivation(_ continuation: CheckedContinuation<Bool, Never>) {
        player?.stop()
        engine?.stop()
        AVAudioSession.sharedInstance().deactivate(options: [.notifyOthersOnDeactivation]) { _, _ in
            continuation.resume(returning: false)
        }
    }

    func deactivate() async {
        await withCheckedContinuation { continuation in
            queue.async {
                self.currentRequest = nil
                self.currentCue = nil
                self.player?.stop()
                self.engine?.stop()
                AVAudioSession.sharedInstance().deactivate(options: [.notifyOthersOnDeactivation]) { _, error in
                    if let error { Self.logger.error("Audio deactivation failed: \(error.localizedDescription, privacy: .public)") }
                    continuation.resume()
                }
            }
        }
    }

    func play(_ cue: DeviceSound, request: AudioPlaybackRequest) {
        queue.async {
            guard !request.isCancelled else { return }
            let interval = DexPerformance.begin("Audio Chime Start")
            defer { DexPerformance.end("Audio Chime Start", interval) }
            do {
                try self.startEngine()
                guard !request.isCancelled, let player = self.player, let buffer = self.buffers[cue] else { return }
                try self.schedule(buffer, on: player, cue: cue, request: request)
            } catch {
                Self.logger.error("Chime playback failed: \(error.localizedDescription, privacy: .public)")
            }
        }
    }

    func playCry(dexID: Int, request: AudioPlaybackRequest, completion: @escaping @Sendable () -> Void) {
        queue.async {
            guard !request.isCancelled else { return }
            do {
                try self.startEngine()
                guard let player = self.player else { throw PlaybackError.invalidFormat }
                let buffer = try Self.cryBuffer(dexID: dexID)
                guard !request.isCancelled else { return }
                try self.schedule(buffer, on: player, request: request, completion: completion)
            } catch {
                Self.logger.error("Cry playback failed for dex \(dexID): \(error.localizedDescription, privacy: .public)")
                // A missing or unreadable asset must not prevent the entry reading.
                if !request.isCancelled { completion() }
            }
        }
    }

    /// Files are already mono 22,050 Hz PCM. Decode on the audio queue and retain
    /// only the scheduled buffer, without loading the whole catalog into memory.
    static func cryBuffer(dexID: Int, bundle: Bundle = .main) throws -> AVAudioPCMBuffer {
        guard let url = bundle.url(forResource: String(format: "pokemon-cry-%03d", dexID), withExtension: "wav") else {
            throw PlaybackError.missingCry
        }
        let file = try AVAudioFile(forReading: url)
        guard file.processingFormat.sampleRate == DeviceSoundWave.sampleRate,
              file.processingFormat.channelCount == 1,
              file.length > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw PlaybackError.invalidFormat
        }
        try file.read(into: buffer)
        return buffer
    }

    private func schedule(_ buffer: AVAudioPCMBuffer, on player: AVAudioPlayerNode, cue: DeviceSound? = nil,
                          request: AudioPlaybackRequest, completion: (@Sendable () -> Void)? = nil) throws {
        player.stop() // Discard previous audio; keep the engine running.
        currentRequest = request
        currentCue = cue
        player.scheduleBuffer(buffer, atTime: nil, options: cue == .scan ? .loops : [], completionCallbackType: .dataPlayedBack) { [weak self] in
            guard let self else { return }
            self.queue.async {
                guard self.currentRequest === request else { return }
                self.currentRequest = nil
                self.currentCue = nil
                if !request.isCancelled { completion?() }
            }
        }
        try player.playAudio()
    }

    func stopSound() {
        queue.async {
            self.currentRequest = nil
            self.currentCue = nil
            self.player?.stop()
        }
    }

    func speak(_ text: String, languageCode: String, request: AudioPlaybackRequest) {
        queue.async {
            guard !request.isCancelled else { return }
            let interval = DexPerformance.begin("Audio Speech Startup")
            defer { DexPerformance.end("Audio Speech Startup", interval) }
            let synthesizer: AVSpeechSynthesizer
            if let existing = self.synthesizer {
                synthesizer = existing
                synthesizer.stopSpeaking(at: .immediate)
            } else {
                synthesizer = AVSpeechSynthesizer()
                synthesizer.usesApplicationAudioSession = true
                self.synthesizer = synthesizer
            }
            let voice = self.voices[languageCode] ?? AVSpeechSynthesisVoice(language: languageCode)
            if let voice { self.voices[languageCode] = voice }
            let utterance = AVSpeechUtterance(string: text)
            utterance.voice = voice
            utterance.rate = languageCode.hasPrefix("zh") ? 0.43 : 0.46
            utterance.pitchMultiplier = 0.84
            utterance.volume = self.speechVolume
            utterance.preUtteranceDelay = 0.28
            guard !request.isCancelled else { return }
            let observer = SpeechPlaybackObserver(utterance: utterance, request: request)
            self.speechObserver = observer
            synthesizer.delegate = observer
            synthesizer.speak(utterance)
        }
    }

    func stopSpeaking() {
        queue.async { self.synthesizer?.stopSpeaking(at: .immediate) }
    }

    /// A nonblocking snapshot also waits for previously enqueued commands.
    func status() async -> Status {
        await withCheckedContinuation { continuation in
            queue.async {
                let progress = self.rendered.snapshot
                continuation.resume(returning: Status(
                    cachedSoundCount: self.buffers.count,
                    hasSpeechSynthesizer: self.synthesizer != nil,
                    soundIsPlaying: self.currentRequest != nil && (self.player?.isPlaying ?? false),
                    speechIsSpeaking: self.synthesizer?.isSpeaking ?? false,
                    renderedFrames: progress.frames,
                    renderedSignalFrames: progress.signalFrames,
                    outputPorts: AVAudioSession.sharedInstance().currentRoute.outputs.map { $0.portType.rawValue },
                    isOnMainThread: Thread.isMainThread,
                    hasSwiftTask: withUnsafeCurrentTask { $0 != nil }
                ))
            }
        }
    }

    private enum PlaybackError: Error { case invalidFormat, missingCry }

    nonisolated struct Status: Sendable {
        let cachedSoundCount: Int
        let hasSpeechSynthesizer: Bool
        let soundIsPlaying: Bool
        let speechIsSpeaking: Bool
        let renderedFrames: Int
        let renderedSignalFrames: Int
        let outputPorts: [String]
        let isOnMainThread: Bool
        let hasSwiftTask: Bool
    }
}

/// A small tap on the chime mixer confirms hardware output is running. This
/// observes only app playback; it never records microphone input.
nonisolated private final class AudioRenderProgress: @unchecked Sendable {
    private let lock = NSLock()
    private var frames = 0
    private var signalFrames = 0

    func record(_ buffer: AVReadOnlyAudioPCMBuffer) {
        let count = buffer.frameLength
        var hasSignal = false
        if case .float(let channel) = buffer.channelData(0) {
            for index in 0..<count where abs(channel[index]) > 0.0001 {
                hasSignal = true
                break
            }
        }
        lock.withLock {
            frames += count
            if hasSignal { signalFrames += count }
        }
    }

    func reset() { lock.withLock { frames = 0; signalFrames = 0 } }
    var snapshot: (frames: Int, signalFrames: Int) { lock.withLock { (frames, signalFrames) } }
}
