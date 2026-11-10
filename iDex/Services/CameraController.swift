import AVFoundation
import CoreImage
import ImageIO
import Observation
import SwiftUI
import UIKit

enum CameraStatus: Equatable {
    case idle, requesting, ready, denied, unavailable
}

enum CameraError: LocalizedError {
    case notReady, noCamera, configurationFailed, captureFailed, cancelled

    var errorDescription: String? {
        switch self {
        case .notReady: "The camera is not ready."
        case .noCamera: "No camera is available on this device."
        case .configurationFailed: "The camera could not be started."
        case .captureFailed: "The photo could not be captured."
        case .cancelled: "The capture was cancelled."
        }
    }
}

/// Camera ownership stays outside the foldable layout so opening the device never
/// destroys or restarts the capture session. Present `CameraPreview` in a square.
@MainActor @Observable
final class CameraController {
    private(set) var status: CameraStatus = .idle
    private(set) var error: CameraError?
    private(set) var device: AVCaptureDevice?
    /// Zoom relative to the 1× wide lens, for the viewfinder readout.
    private(set) var displayZoom: CGFloat = 1
    @ObservationIgnored private let worker = CameraSessionWorker()
    @ObservationIgnored private var startTask: Task<Void, Never>?
    @ObservationIgnored private var wantsCamera = false
    @ObservationIgnored private var captureRotation: CGFloat = 90
    @ObservationIgnored private var startGeneration = 0

    var session: AVCaptureSession { worker.session }

    func start() {
        wantsCamera = true
        guard startTask == nil, status != .ready else { return }
        status = .requesting
        error = nil
        startGeneration += 1
        let generation = startGeneration
        startTask = Task { [weak self] in
            guard let self else { return }
            let authorized: Bool
            switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized: authorized = true
            case .notDetermined: authorized = await AVCaptureDevice.requestAccess(for: .video)
            default: authorized = false
            }
            guard self.wantsCamera, !Task.isCancelled, generation == self.startGeneration else {
                if generation == self.startGeneration { self.startTask = nil }
                return
            }
            guard authorized else {
                self.status = .denied
                self.startTask = nil
                return
            }
            do {
                let device = try await self.worker.start()
                if self.wantsCamera, !Task.isCancelled, generation == self.startGeneration {
                    self.device = device
                    self.status = .ready
                } else if !self.wantsCamera {
                    self.worker.stop()
                }
            } catch {
                if self.wantsCamera, !Task.isCancelled, generation == self.startGeneration {
                    self.error = (error as? CameraError) ?? .configurationFailed
                    self.status = .unavailable
                }
            }
            if generation == self.startGeneration { self.startTask = nil }
        }
    }

    func stop() {
        wantsCamera = false
        startGeneration += 1
        startTask?.cancel()
        startTask = nil
        worker.stop()
        status = .idle
    }

    /// Retain the latest viewfinder frame before suspending. Moving the camera
    /// after the tap cannot change the image being scanned.
    func capture() async throws -> CGImage {
        guard status == .ready else { throw CameraError.notReady }
        let frame = try worker.snapshot()
        return try await worker.render(frame, rotation: captureRotation)
    }

    fileprivate func updateCaptureRotation(_ angle: CGFloat) {
        captureRotation = angle
    }

    /// Pinch gesture: `scale` multiplies the zoom that was active when the pinch began.
    fileprivate func zoom(to relative: CGFloat) {
        worker.setZoom(relative: relative) { [weak self] applied in
            Task { @MainActor [weak self] in self?.displayZoom = applied }
        }
    }

    fileprivate var currentZoom: CGFloat { displayZoom }

    /// Tap in the preview: `point` is already in capture-device coordinates (0…1).
    fileprivate func focus(at point: CGPoint) {
        worker.focus(at: point)
    }

    fileprivate func connectPreview(_ layer: AVCaptureVideoPreviewLayer, completion: @escaping @MainActor @Sendable () -> Void) {
        worker.connectPreview(layer, completion: completion)
    }

    fileprivate func disconnectPreview(_ layer: AVCaptureVideoPreviewLayer) {
        worker.disconnectPreview(layer)
    }
}

/// Retain just one sensor frame. Rendering happens only on a scan, away from the
/// UI actor; there is no still-photo shutter, autofocus, or encoding delay.
nonisolated private final class CameraSessionWorker: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    let session: AVCaptureSession = {
        let session = AVCaptureSession()
        // This session captures video only. Audio playback owns the shared
        // session; camera startup must not change its category or routing.
        session.automaticallyConfiguresApplicationAudioSession = false
        return session
    }()
    private let queue = DispatchQueue(label: "app.idex.camera", qos: .userInitiated)
    private let frameQueue = DispatchQueue(label: "app.idex.camera.frames", qos: .userInitiated)
    private let videoOutput = AVCaptureVideoDataOutput()
    private let frames = CameraFrameBuffer()
    private let imageContext = CIContext(options: [.cacheIntermediates: false])
    private var configured = false
    private var device: AVCaptureDevice?
    /// Video zoom factor of the 1× wide lens on the chosen (possibly virtual) device.
    private var wideZoomFactor: CGFloat = 1
    private var pendingStart: CheckedContinuation<AVCaptureDevice, Error>?

    func start() async throws -> AVCaptureDevice {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    if !self.configured { try self.configure() }
                    self.frameQueue.sync { self.frames.clear() }
                    self.pendingStart = continuation
                    if !self.session.isRunning { self.session.startRunning() }
                    guard self.session.isRunning else { throw CameraError.configurationFailed }
                } catch {
                    self.pendingStart = nil
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func configure() throws {
        // Prefer the multi-lens virtual device: it switches to the ultra-wide
        // lens for close-ups (macro) and back on its own, and exposes the full
        // optical zoom range through a single videoZoomFactor.
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInTripleCamera, .builtInDualWideCamera, .builtInDualCamera, .builtInWideAngleCamera],
            mediaType: .video, position: .back)
        guard let device = discovery.devices.first ?? AVCaptureDevice.default(for: .video) else { throw CameraError.noCamera }
        let input = try AVCaptureDeviceInput(device: device)
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .high
        guard session.canAddInput(input), session.canAddOutput(videoOutput) else {
            throw CameraError.configurationFailed
        }
        session.addInput(input)
        configureFocusAndZoom(device)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: frameQueue)
        session.addOutput(videoOutput)
        // Retain native YUV frames; Core Image converts only the tapped frame.
        // BGRA requires AVFoundation to convert every frame in the live feed.
        let availableFormats = videoOutput.availableVideoPixelFormatTypes
        let preferredFormats = [kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                                kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                                kCVPixelFormatType_32BGRA]
        if let format = preferredFormats.first(where: availableFormats.contains) {
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: format]
        }
        // Keep sensor pixels unrotated; apply the tap-time capture angle once.
        if let connection = videoOutput.connection(with: .video), connection.isVideoRotationAngleSupported(0) {
            connection.videoRotationAngle = 0
        }
        self.device = device
        configured = true
    }

    /// Start at the 1× wide lens, continuous focus/exposure that re-triggers when
    /// the subject changes, and no near/far focus restriction so both close toys
    /// and distant screens lock quickly.
    private func configureFocusAndZoom(_ device: AVCaptureDevice) {
        wideZoomFactor = device.virtualDeviceSwitchOverVideoZoomFactors.first.map { CGFloat(truncating: $0) } ?? 1
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            device.videoZoomFactor = min(max(wideZoomFactor, device.minAvailableVideoZoomFactor), device.maxAvailableVideoZoomFactor)
            if device.isFocusModeSupported(.continuousAutoFocus) { device.focusMode = .continuousAutoFocus }
            if device.isExposureModeSupported(.continuousAutoExposure) { device.exposureMode = .continuousAutoExposure }
            if device.isAutoFocusRangeRestrictionSupported { device.autoFocusRangeRestriction = .none }
            if device.isSmoothAutoFocusSupported { device.isSmoothAutoFocusEnabled = false }
            device.isSubjectAreaChangeMonitoringEnabled = true
        } catch {
            // Defaults still work; zoom and focus simply stay at the system values.
        }
    }

    /// `relative` is the zoom shown to the user (1 = wide lens). Clamped to the
    /// device range, allowing the ultra-wide (below 1) and up to 8× digital.
    func setZoom(relative: CGFloat, applied: @escaping @Sendable (CGFloat) -> Void) {
        queue.async {
            guard let device = self.device else { return }
            let lower = device.minAvailableVideoZoomFactor
            let upper = min(device.maxAvailableVideoZoomFactor, self.wideZoomFactor * 8)
            let factor = min(max(relative * self.wideZoomFactor, lower), upper)
            do {
                try device.lockForConfiguration()
                device.videoZoomFactor = factor
                device.unlockForConfiguration()
            } catch { return }
            applied(factor / self.wideZoomFactor)
        }
    }

    func focus(at point: CGPoint) {
        queue.async {
            guard let device = self.device else { return }
            do {
                try device.lockForConfiguration()
                defer { device.unlockForConfiguration() }
                if device.isFocusPointOfInterestSupported, device.isFocusModeSupported(.continuousAutoFocus) {
                    device.focusPointOfInterest = point
                    device.focusMode = .continuousAutoFocus
                }
                if device.isExposurePointOfInterestSupported, device.isExposureModeSupported(.continuousAutoExposure) {
                    device.exposurePointOfInterest = point
                    device.exposureMode = .continuousAutoExposure
                }
                device.isSubjectAreaChangeMonitoringEnabled = true
            } catch {}
        }
    }

    func stop() {
        queue.async {
            if self.session.isRunning { self.session.stopRunning() }
            self.frameQueue.sync { self.frames.clear() }
            if let pending = self.pendingStart {
                self.pendingStart = nil
                pending.resume(throwing: CameraError.cancelled)
            }
        }
    }

    // Assigning a preview's session changes the capture graph and may block.
    // Serialize it with start/stop instead of doing it during SwiftUI layout or
    // dismantling on the main thread. Layer geometry still belongs to UIKit.
    func connectPreview(_ layer: AVCaptureVideoPreviewLayer, completion: @escaping @MainActor @Sendable () -> Void) {
        let target = CameraPreviewTarget(layer: layer)
        queue.async {
            let interval = DexPerformance.begin("Camera Preview Connect")
            target.layer.session = self.session
            DexPerformance.end("Camera Preview Connect", interval)
            Task { @MainActor in completion() }
        }
    }

    func disconnectPreview(_ layer: AVCaptureVideoPreviewLayer) {
        let target = CameraPreviewTarget(layer: layer)
        queue.async {
            let interval = DexPerformance.begin("Camera Preview Disconnect")
            target.layer.session = nil
            DexPerformance.end("Camera Preview Disconnect", interval)
        }
    }

    func snapshot() throws -> CameraFrame {
        try frames.snapshot()
    }

    func render(_ frame: CameraFrame, rotation: CGFloat) async throws -> CGImage {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                guard let result = CameraFrameRenderer.squareImage(
                    CIImage(cvPixelBuffer: frame.pixels), rotation: rotation, context: self.imageContext
                ) else {
                    continuation.resume(throwing: CameraError.captureFailed)
                    return
                }
                continuation.resume(returning: result)
            }
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixels = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        if frames.update(pixels) {
            queue.async {
                guard let pending = self.pendingStart else { return }
                self.pendingStart = nil
                if let device = self.device { pending.resume(returning: device) }
                else { pending.resume(throwing: CameraError.noCamera) }
            }
        }
    }
}

/// Only the capture queue changes the layer's session. UIKit retains ownership
/// of its bounds and presentation; retaining the target also keeps teardown off
/// the main thread after SwiftUI removes the preview view.
nonisolated private struct CameraPreviewTarget: @unchecked Sendable {
    let layer: AVCaptureVideoPreviewLayer
}

// Captures own their retained buffer; replacing or clearing the live frame
// cannot mutate a snapshot already handed to recognition.
nonisolated struct CameraFrame: @unchecked Sendable {
    let pixels: CVPixelBuffer
}

nonisolated final class CameraFrameBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var latest: CameraFrame?

    @discardableResult func update(_ pixels: CVPixelBuffer) -> Bool {
        lock.withLock {
            let isFirstFrame = latest == nil
            latest = CameraFrame(pixels: pixels)
            return isFirstFrame
        }
    }

    func clear() {
        lock.withLock { latest = nil }
    }

    func snapshot() throws -> CameraFrame {
        try lock.withLock {
            guard let latest else { throw CameraError.notReady }
            return latest
        }
    }
}

nonisolated enum CameraFrameRenderer {
    static func squareImage(_ image: CIImage, rotation: CGFloat, context: CIContext) -> CGImage? {
        // AVFoundation angles are clockwise; Core Image coordinates point up.
        let upright = image.transformed(by: CGAffineTransform(rotationAngle: -rotation * .pi / 180))
        let extent = upright.extent
        let side = floor(min(extent.width, extent.height))
        guard side > 0 else { return nil }
        let crop = CGRect(x: floor(extent.midX - side / 2), y: floor(extent.midY - side / 2), width: side, height: side)
        return context.createCGImage(upright, from: crop)
    }
}

struct CameraPreview: UIViewRepresentable {
    let controller: CameraController

    func makeUIView(context: Context) -> CameraPreviewView {
        let view = CameraPreviewView()
        view.onCaptureRotation = { [weak controller] in controller?.updateCaptureRotation($0) }
        view.updateCamera(controller: controller)
        return view
    }

    func updateUIView(_ uiView: CameraPreviewView, context: Context) {
        uiView.updateCamera(controller: controller)
    }

    static func dismantleUIView(_ uiView: CameraPreviewView, coordinator: ()) {
        uiView.disconnectCamera()
    }
}

@MainActor final class CameraPreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    var onCaptureRotation: ((CGFloat) -> Void)?
    private var controller: CameraController?
    private var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private var previewRotationObservation: NSKeyValueObservation?
    private var captureRotationObservation: NSKeyValueObservation?

    private var pinchStartZoom: CGFloat = 1

    override init(frame: CGRect) {
        super.init(frame: frame)
        previewLayer.videoGravity = .resizeAspectFill
        backgroundColor = .black
        isAccessibilityElement = true
        accessibilityIdentifier = "camera.preview"
        addGestureRecognizer(UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:))))
        addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(handleTap(_:))))
    }

    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        guard let controller else { return }
        switch gesture.state {
        case .began: pinchStartZoom = controller.currentZoom
        case .changed: controller.zoom(to: pinchStartZoom * gesture.scale)
        default: break
        }
    }

    @objc private func handleTap(_ gesture: UITapGestureRecognizer) {
        guard let controller else { return }
        let point = previewLayer.captureDevicePointConverted(fromLayerPoint: gesture.location(in: self))
        controller.focus(at: point)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func layoutSubviews() {
        super.layoutSubviews()
        updateRotation()
    }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        updateRotation()
    }

    func updateCamera(controller: CameraController) {
        if self.controller !== controller {
            self.controller = controller
            controller.connectPreview(previewLayer) { [weak self, weak controller] in
                guard let self, self.controller === controller else { return }
                self.updateRotation()
            }
        }
        if let device = controller.device, rotationCoordinator?.device !== device {
            previewRotationObservation = nil
            captureRotationObservation = nil
            let coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: previewLayer)
            rotationCoordinator = coordinator
            // AVFoundation delivers both observations on the main queue. Apply
            // them immediately, including display changes with no UI rotation.
            previewRotationObservation = coordinator.observe(\.videoRotationAngleForHorizonLevelPreview, options: [.initial, .new]) { [weak self] _, _ in
                MainActor.assumeIsolated { self?.updateRotation() }
            }
            captureRotationObservation = coordinator.observe(\.videoRotationAngleForHorizonLevelCapture, options: [.initial, .new]) { [weak self] _, _ in
                MainActor.assumeIsolated { self?.updateRotation() }
            }
        }
        updateRotation()
    }

    func disconnectCamera() {
        previewRotationObservation = nil
        captureRotationObservation = nil
        rotationCoordinator = nil
        onCaptureRotation = nil
        controller?.disconnectPreview(previewLayer)
        controller = nil
    }

    func updateRotation() {
        guard let coordinator = rotationCoordinator else { return }
        // Duo reports .portrait on both displays, but their native rotations
        // differ. The coordinator accounts for the camera and preview layer.
        let rotation = coordinator.videoRotationAngleForHorizonLevelPreview
        if let connection = previewLayer.connection,
           connection.videoRotationAngle != rotation,
           connection.isVideoRotationAngleSupported(rotation) {
            connection.videoRotationAngle = rotation
        }
        // Photo rotation is relative to the sensor, independently of the layer.
        onCaptureRotation?(coordinator.videoRotationAngleForHorizonLevelCapture)
    }
}
