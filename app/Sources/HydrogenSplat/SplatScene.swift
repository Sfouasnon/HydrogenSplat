import SwiftUI
import AppKit
import Metal
import MetalKit
import simd
import HSCore
import MetalSplatter
import SplatIO

// MARK: - Can this build render at all?

/// MetalSplatter finds its compiled shaders with `device.makeDefaultLibrary(bundle: .module)` and
/// calls fatalError when that fails; it also force-unwraps its resource bundle's identifier.
/// `swift build` compiles no Metal and writes no Info.plist into resource bundles, so a
/// SwiftPM-built binary would trap on the first model. xcodebuild does both (make_app.sh uses
/// it). Look before leaping: the viewer is offered only when the bundle it needs is there.
enum SplatViewerSupport {
    static let bundleName = "MetalSplatter_MetalSplatter.bundle"

    /// nil when the viewer can run; otherwise why not, for the user.
    static let problem: String? = {
        #if arch(x86_64)
        return "The splat viewer needs Apple silicon (MetalSplatter does not support Intel)."
        #else
        guard MTLCreateSystemDefaultDevice() != nil else { return "No Metal device." }
        // the places the generated Bundle.module accessor looks, in its order
        let roots = [Bundle.main.resourceURL, Bundle.main.bundleURL,
                     Bundle.main.executableURL?.deletingLastPathComponent()]
        for case let root? in roots {
            let url = root.appendingPathComponent(bundleName)
            guard let b = Bundle(url: url) else { continue }
            if b.bundleIdentifier != nil, b.url(forResource: "default", withExtension: "metallib") != nil {
                return nil
            }
            return "This build has no compiled Metal shaders (\(bundleName) holds no default.metallib). "
                 + "`swift build` / `swift run` cannot compile them — build the app with app/scripts/make_app.sh."
        }
        return "\(bundleName) is missing from this build — build the app with app/scripts/make_app.sh."
        #endif
    }()
}

// MARK: - The view that takes the mouse

final class SplatMTKView: MTKView {
    var onOrbit: ((CGFloat, CGFloat) -> Void)?       // points dragged
    var onPan: ((CGFloat, CGFloat) -> Void)?
    var onZoom: ((CGFloat) -> Void)?                 // multiplicative: < 1 moves in

    override var acceptsFirstResponder: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func mouseDragged(with e: NSEvent) {
        if e.modifierFlags.contains(.shift) { onPan?(e.deltaX, e.deltaY) } else { onOrbit?(e.deltaX, e.deltaY) }
    }
    override func rightMouseDragged(with e: NSEvent) { onPan?(e.deltaX, e.deltaY) }
    override func otherMouseDragged(with e: NSEvent) { onPan?(e.deltaX, e.deltaY) }

    override func scrollWheel(with e: NSEvent) {
        // trackpads report fine pixel deltas, wheels coarse line deltas
        let d = e.hasPreciseScrollingDeltas ? e.scrollingDeltaY * 0.004 : e.scrollingDeltaY * 0.05
        onZoom?(CGFloat(exp(-Double(d))))
    }
    override func magnify(with e: NSEvent) { onZoom?(1 / max(0.2, 1 + e.magnification)) }
}

// MARK: - The scene

@MainActor
final class SplatScene: NSObject, ObservableObject, MTKViewDelegate {
    enum Phase: Equatable {
        case idle
        case loading(String)
        case ready
        case failed(String)
    }

    enum CameraMode: Equatable {
        /// Exactly a capture's or a move frame's camera: its pose and its pinhole.
        case pose(ViewerPose, label: String)
        case orbit
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var splatCount = 0
    @Published private(set) var loadSeconds: Double = 0
    @Published var mode: CameraMode = .orbit
    @Published private(set) var cameras: CameraSet?

    var orbit = OrbitCamera(eye: SIMD3(0, 0, -1), target: .zero, up: SIMD3(0, -1, 0), fovy: 50 * .pi / 180)

    let device: MTLDevice?
    private let queue: MTLCommandQueue?
    private var renderer: SplatRenderer?
    private var drawableSize: CGSize = .zero
    private static let maxRenders = 3
    private let inFlight = DispatchSemaphore(value: SplatScene.maxRenders)
    private var loadTask: Task<Void, Never>?

    override init() {
        device = MTLCreateSystemDefaultDevice()
        queue = device?.makeCommandQueue()
        super.init()
    }

    func configure(_ view: SplatMTKView) {
        view.device = device
        view.colorPixelFormat = .bgra8Unorm_srgb
        view.depthStencilPixelFormat = .depth32Float
        view.sampleCount = 1
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)   // Brush trains against black
        view.preferredFramesPerSecond = 60
        view.delegate = self
        // weak view: the view owns these closures, so a strong capture would keep it alive forever
        view.onOrbit = { [weak self, weak view] dx, dy in self?.dragOrbit(dx, dy, viewHeight: view?.bounds.height ?? 1) }
        view.onPan = { [weak self, weak view] dx, dy in self?.dragPan(dx, dy, viewHeight: view?.bounds.height ?? 1) }
        view.onZoom = { [weak self] f in self?.zoom(f) }
        drawableSize = view.drawableSize
    }

    // MARK: loading

    func load(model: ViewerModelFile, config: EngineConfig) {
        loadTask?.cancel()
        renderer = nil
        splatCount = 0
        if let why = SplatViewerSupport.problem { phase = .failed(why); return }
        guard let device = device else { phase = .failed("No Metal device."); return }
        phase = .loading("asking the engine for the cameras…")
        loadTask = Task { [weak self] in
            let t0 = Date()
            do {
                // cameras first: they are quick, and a model with no frame is not worth the wait
                let cams = try await CameraExport.run(config: config, model: model)
                guard let self = self, !Task.isCancelled else { return }
                self.cameras = cams
                self.resetOrbit()
                self.phase = .loading("reading \((model.ply as NSString).lastPathComponent) (\(model.sizeLabel))…")

                let r = try SplatRenderer(device: device, colorFormat: .bgra8Unorm_srgb, depthFormat: .depth32Float,
                                          sampleCount: 1, maxViewCount: 1, maxSimultaneousRenders: SplatScene.maxRenders,
                                          clearColor: MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1))
                let url = URL(fileURLWithPath: model.ply)
                let chunk = try await Task.detached(priority: .userInitiated) { () async throws -> SplatChunk in
                    let points = try await AutodetectSceneReader(url).readAll()
                    return try SplatChunk(device: device, from: points)
                }.value
                guard !Task.isCancelled else { return }
                self.phase = .loading("sorting \(chunk.splatCount.formatted()) splats…")
                _ = await r.addChunk(chunk)
                guard !Task.isCancelled else { return }
                self.renderer = r
                self.splatCount = chunk.splatCount
                self.loadSeconds = Date().timeIntervalSince(t0)
                self.phase = .ready
            } catch {
                guard let self = self, !Task.isCancelled else { return }
                self.phase = .failed(error.localizedDescription)
            }
        }
    }

    func unload() {
        loadTask?.cancel()
        renderer = nil
    }

    // MARK: camera

    /// Start where the first capture stood, turning about the subject.
    func resetOrbit() {
        guard let c = cameras else { return }
        if let p = c.views.first?.pose {
            orbit = OrbitCamera(leaving: p, subject: c.subject, up: c.up)
            orbit.target = c.subject
        } else {
            orbit = OrbitCamera(eye: c.subject - SIMD3(0, 0, 1), target: c.subject, up: c.up, fovy: 50 * .pi / 180)
        }
        mode = .orbit
    }

    func stand(at camera: ViewerCamera) {
        guard let p = camera.pose else { return }
        mode = .pose(p, label: camera.label)
    }

    func stand(at pose: ViewerPose, label: String) { mode = .pose(pose, label: label) }

    /// The first drag leaves a capture's pose from where it stands, so nothing jumps.
    private func enterOrbitIfNeeded() {
        if case .pose(let p, _) = mode, let c = cameras {
            orbit = OrbitCamera(leaving: p, subject: c.subject, up: c.up)
            mode = .orbit
        }
    }

    private func dragOrbit(_ dx: CGFloat, _ dy: CGFloat, viewHeight: CGFloat) {
        enterOrbitIfNeeded()
        let k = Float(Double.pi / max(Double(viewHeight), 1))      // a full-height drag is half a turn
        orbit.rotate(yaw: -Float(dx) * k, pitch: Float(dy) * k)
    }

    private func dragPan(_ dx: CGFloat, _ dy: CGFloat, viewHeight: CGFloat) {
        enterOrbitIfNeeded()
        let h = Float(max(viewHeight, 1))
        orbit.pan(dx: Float(dx) / h, dy: Float(dy) / h)
    }

    private func zoom(_ factor: CGFloat) {
        enterOrbitIfNeeded()
        orbit.dolly(factor: Float(factor))
    }

    private var matrices: (projection: simd_float4x4, view: simd_float4x4) {
        let w = Float(drawableSize.width), h = Float(drawableSize.height)
        switch mode {
        case .pose(let p, _):
            return (ViewerMath.projection(pose: p, drawableWidth: w, drawableHeight: h), ViewerMath.viewMatrix(c2w: p.c2w))
        case .orbit:
            return (ViewerMath.perspective(fovy: orbit.fovy, aspect: w / max(h, 1)), orbit.viewMatrix)
        }
    }

    // MARK: MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) { drawableSize = size }

    func draw(in view: MTKView) {
        guard let renderer = renderer, renderer.isReadyToRender, let queue = queue,
              drawableSize.width > 0, drawableSize.height > 0,
              let drawable = view.currentDrawable else { return }
        _ = inFlight.wait(timeout: .distantFuture)
        guard let cb = queue.makeCommandBuffer() else { inFlight.signal(); return }
        let sem = inFlight
        cb.addCompletedHandler { _ in sem.signal() }

        let m = matrices
        let vp = SplatRenderer.ViewportDescriptor(
            viewport: MTLViewport(originX: 0, originY: 0, width: Double(drawableSize.width), height: Double(drawableSize.height), znear: 0, zfar: 1),
            projectionMatrix: m.projection, viewMatrix: m.view,
            screenSize: SIMD2(Int(drawableSize.width), Int(drawableSize.height)))
        let drew = (try? renderer.render(viewports: [vp], colorTexture: drawable.texture, colorStoreAction: .store,
                                         depthTexture: view.depthStencilTexture, rasterizationRateMap: nil,
                                         renderTargetArrayLength: 0, to: cb)) ?? false
        if drew { cb.present(drawable) }
        cb.commit()
    }
}

/// SwiftUI wrapper. The scene owns the state; this only hands it a view.
struct SplatMetalView: NSViewRepresentable {
    let scene: SplatScene

    func makeNSView(context: Context) -> SplatMTKView {
        let v = SplatMTKView()
        scene.configure(v)
        return v
    }

    func updateNSView(_ nsView: SplatMTKView, context: Context) {}
}
