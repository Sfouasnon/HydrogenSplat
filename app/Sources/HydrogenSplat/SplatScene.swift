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
    /// The model this scene holds (or is loading); nil when empty.
    @Published private(set) var loaded: ViewerModelFile?
    /// The capture picker's state lives here, not in the view, so leaving the Viewer page and
    /// coming back finds the camera where it was instead of snapping to the first capture.
    @Published var capture = ""
    @Published var eye = "L"
    /// A text field beside the viewer (the move script) has the keyboard: the viewer's
    /// single-key shortcuts stand down so typing a space does not start playback.
    @Published var typing = false

    /// Set while a keyframed move is open: the free camera turns about this point (metres) and
    /// looks through `lockLens`, level — exactly how a baked frame is built — so what is keyed
    /// is what renders. Pan is off (it would aim the view somewhere a frame cannot).
    @Published var anchorLock: SIMD3<Float>?
    var lockLens: KeyedMove.Lens?

    /// The camera move being played or scrubbed (a built move or the move panel's live preview).
    let playback = MovePlayback()

    var orbit = OrbitCamera(eye: SIMD3(0, 0, -1), target: .zero, up: SIMD3(0, -1, 0), fovy: 50 * .pi / 180)

    // MARK: live grade

    /// The look being previewed: the same per-channel 256-entry tables `hs grade` bakes with ffmpeg
    /// lutrgb, applied to the rendered 8-bit code values in a second pass. nil draws the model as-is.
    @Published var look: GradeSettings?
    /// Set when the grade pass could not be built; the viewer then draws ungraded and says so.
    @Published var lookProblem: String?
    private var offscreen: MTLTexture?
    private var offscreenCodes: MTLTexture?        // the same pixels viewed as bgra8Unorm: raw code values
    private var lutTexture: MTLTexture?
    private var lutFor: GradeSettings?
    private var gradePipeline: MTLRenderPipelineState?
    private var gradeBuildFailed = false

    private struct GradePass {
        let source: MTLTexture, codes: MTLTexture, lut: MTLTexture, pipe: MTLRenderPipelineState
    }

    /// Full-screen triangle; the fragment looks each code value up in the grade tables and hands the
    /// sRGB drawable the linear value that stores exactly that code — so the preview is the curve
    /// ffmpeg applies to code values, not an approximation of it.
    private static let gradeShader = """
    #include <metal_stdlib>
    using namespace metal;
    struct VOut { float4 pos [[position]]; float2 uv; };
    vertex VOut grade_vs(uint vid [[vertex_id]]) {
        float2 p = float2(float((vid << 1) & 2), float(vid & 2));
        VOut o; o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0); o.uv = float2(p.x, 1.0 - p.y); return o;
    }
    fragment float4 grade_fs(VOut in [[stage_in]], texture2d<float> src [[texture(0)]],
                             texture2d<float> lut [[texture(1)]]) {
        constexpr sampler s(filter::nearest);
        float4 c = src.sample(s, in.uv);
        float3 o;
        for (int i = 0; i < 3; i++) {
            uint k = uint(clamp(c[i], 0.0, 1.0) * 255.0 + 0.5);
            o[i] = lut.read(uint2(k, 0))[i];
        }
        o = select(pow((o + 0.055) / 1.055, 2.4), o / 12.92, o <= 0.04045);
        return float4(o, 1.0);
    }
    """

    private func prepareGrade(_ g: GradeSettings, width: Int, height: Int) -> GradePass? {
        guard let device = device, !gradeBuildFailed else { return nil }
        if gradePipeline == nil {
            do {
                let lib = try device.makeLibrary(source: SplatScene.gradeShader, options: nil)
                let d = MTLRenderPipelineDescriptor()
                d.vertexFunction = lib.makeFunction(name: "grade_vs")
                d.fragmentFunction = lib.makeFunction(name: "grade_fs")
                d.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
                gradePipeline = try device.makeRenderPipelineState(descriptor: d)
            } catch {
                gradeBuildFailed = true
                let why = error.localizedDescription
                DispatchQueue.main.async {
                    self.lookProblem = "The live grade could not be built (\(why)); the viewer shows the model ungraded."
                }
                return nil
            }
        }
        if offscreen == nil || offscreen!.width != width || offscreen!.height != height {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm_srgb, width: width,
                                                             height: height, mipmapped: false)
            d.usage = [.renderTarget, .shaderRead, .pixelFormatView]
            d.storageMode = .private
            offscreen = device.makeTexture(descriptor: d)
            offscreenCodes = offscreen?.makeTextureView(pixelFormat: .bgra8Unorm)
        }
        if lutTexture == nil || lutFor != g {
            // a fresh texture per change: frames still in flight keep reading the one they had
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 256, height: 1,
                                                             mipmapped: false)
            d.usage = [.shaderRead]
            let r = g.lut(0), gg = g.lut(1), b = g.lut(2)
            var bytes = [UInt8](repeating: 255, count: 256 * 4)
            for i in 0..<256 { bytes[i * 4] = r[i]; bytes[i * 4 + 1] = gg[i]; bytes[i * 4 + 2] = b[i] }
            let t = device.makeTexture(descriptor: d)
            t?.replace(region: MTLRegionMake2D(0, 0, 256, 1), mipmapLevel: 0, withBytes: bytes, bytesPerRow: 256 * 4)
            lutTexture = t
            lutFor = g
        }
        guard let src = offscreen, let codes = offscreenCodes, let lut = lutTexture, let pipe = gradePipeline
        else { return nil }
        return GradePass(source: src, codes: codes, lut: lut, pipe: pipe)
    }

    private func encodeGrade(_ g: GradePass, into target: MTLTexture, cb: MTLCommandBuffer) {
        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = target
        rp.colorAttachments[0].loadAction = .dontCare
        rp.colorAttachments[0].storeAction = .store
        guard let enc = cb.makeRenderCommandEncoder(descriptor: rp) else { return }
        enc.setRenderPipelineState(g.pipe)
        enc.setFragmentTexture(g.codes, index: 0)
        enc.setFragmentTexture(g.lut, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
    }

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
        playback.scene = self
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
        cameras = nil
        capture = ""
        loaded = model
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
        playback.clear()
        loadTask?.cancel()
        renderer = nil
        splatCount = 0
        cameras = nil
        capture = ""
        loaded = nil
        phase = .idle
    }

    // MARK: camera

    /// Where the viewer's camera is, in solve coordinates (mm) — what a move's start line is
    /// located against. Only the position counts: every move frame looks at the subject.
    var cameraPositionMM: SIMD3<Double> {
        let p: SIMD3<Float>
        switch mode {
        case .pose(let pose, _): p = pose.position
        case .orbit: p = orbit.eye
        }
        return SIMD3<Double>(Double(p.x), Double(p.y), Double(p.z)) * 1000
    }

    /// Start where the first capture stood, turning about the subject.
    func resetOrbit() {
        guard let c = cameras else { return }
        if let a = anchorLock {
            // keep the position, face the anchor through the move's lens
            orbit = OrbitCamera(eye: cameraPosition, target: a, up: c.up,
                                fovy: Float(lockLens?.verticalFOV ?? Double(orbit.fovy)))
            playback.pause()
            mode = .orbit
            return
        }
        if let p = c.views.first?.pose {
            orbit = OrbitCamera(leaving: p, subject: c.subject, up: c.up)
            orbit.target = c.subject
        } else {
            orbit = OrbitCamera(eye: c.subject - SIMD3(0, 0, 1), target: c.subject, up: c.up, fovy: 50 * .pi / 180)
        }
        playback.pause()
        mode = .orbit
    }

    func stand(at camera: ViewerCamera) {
        guard let p = camera.pose else { return }
        playback.pause()
        mode = .pose(p, label: camera.label)
    }

    func stand(at pose: ViewerPose, label: String) {
        playback.pause()
        mode = .pose(pose, label: label)
    }

    /// Playback's own route: does not pause the playback that called it.
    func stand(atMoveFrame pose: ViewerPose, label: String) { mode = .pose(pose, label: label) }

    /// The first drag leaves a capture's pose from where it stands, so nothing jumps.
    private func enterOrbitIfNeeded() {
        if case .pose(let p, _) = mode, let c = cameras {
            playback.pause()
            if let a = anchorLock {
                orbit = OrbitCamera(eye: p.position, target: a, up: c.up, fovy: p.verticalFOV)
                mode = .orbit
                return
            }
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
        guard anchorLock == nil else { return }
        enterOrbitIfNeeded()
        let h = Float(max(viewHeight, 1))
        orbit.pan(dx: Float(dx) / h, dy: Float(dy) / h)
    }

    private func zoom(_ factor: CGFloat) {
        enterOrbitIfNeeded()
        orbit.dolly(factor: Float(factor))
    }

    private var matrices: (projection: simd_float4x4, view: simd_float4x4) {
        matrices(width: Float(drawableSize.width), height: Float(drawableSize.height))
    }

    /// The pose the free camera stands at while a move is open: the orbit eye, looking at the
    /// anchor through the move's lens, level.
    private var lockedPose: ViewerPose? {
        guard let a = anchorLock, let lens = lockLens, let c = cameras else { return nil }
        let e = orbit.eye, u = c.up
        let m = KeyedMove.levelCameraToWorld(eye: SIMD3(Double(e.x), Double(e.y), Double(e.z)),
                                             target: SIMD3(Double(a.x), Double(a.y), Double(a.z)),
                                             up: SIMD3(Double(u.x), Double(u.y), Double(u.z)))
        let f = simd_float4x4(columns: (SIMD4<Float>(m.columns.0), SIMD4<Float>(m.columns.1),
                                        SIMD4<Float>(m.columns.2), SIMD4<Float>(m.columns.3)))
        return ViewerPose(c2w: f, w: Float(lens.w), h: Float(lens.h), fx: Float(lens.fx), fy: Float(lens.fy),
                          cx: Float(lens.cx), cy: Float(lens.cy))
    }

    /// The pinhole on screen right now, if the view is one (a capture, a move frame, or the
    /// locked free camera): what the crop guides are drawn around.
    var screenPose: ViewerPose? {
        switch mode {
        case .pose(let p, _): return p
        case .orbit: return lockedPose
        }
    }

    func matrices(width w: Float, height h: Float) -> (projection: simd_float4x4, view: simd_float4x4) {
        if let p = screenPose {
            return (ViewerMath.projection(pose: p, drawableWidth: w, drawableHeight: h), ViewerMath.viewMatrix(c2w: p.c2w))
        }
        return (ViewerMath.perspective(fovy: orbit.fovy, aspect: w / max(h, 1)), orbit.viewMatrix)
    }

    /// A world point (metres) on a view of this size (points), or nil behind the camera.
    func screenPoint(_ p: SIMD3<Float>, size: CGSize) -> CGPoint? {
        let m = matrices(width: Float(size.width), height: Float(size.height))
        let c = m.projection * m.view * SIMD4<Float>(p, 1)
        guard c.w > 1e-6 else { return nil }
        return CGPoint(x: CGFloat((c.x / c.w + 1) / 2) * size.width, y: CGFloat((1 - c.y / c.w) / 2) * size.height)
    }

    /// The nearest-to-camera sparse point within `radius` points of a click: where a person is.
    func pickPoint(at q: CGPoint, size: CGSize, radius: CGFloat = 14) -> SIMD3<Float>? {
        guard let pts = cameras?.points, !pts.isEmpty else { return nil }
        let m = matrices(width: Float(size.width), height: Float(size.height))
        let pv = m.projection * m.view
        var best: (SIMD3<Float>, Float)?
        for p in pts {
            let c = pv * SIMD4<Float>(p, 1)
            guard c.w > 1e-6 else { continue }
            let x = CGFloat((c.x / c.w + 1) / 2) * size.width, y = CGFloat((1 - c.y / c.w) / 2) * size.height
            let d = hypot(x - q.x, y - q.y)
            guard d <= radius else { continue }
            if best == nil || c.w < best!.1 { best = (p, c.w) }
        }
        return best?.0
    }

    /// Where the viewer's camera is (metres).
    var cameraPosition: SIMD3<Float> {
        switch mode {
        case .pose(let p, _): return p.position
        case .orbit: return orbit.eye
        }
    }

    /// Lock the free camera to a move's anchor and lens (nil unlocks).
    func lock(anchor: SIMD3<Float>?, lens: KeyedMove.Lens?) {
        anchorLock = anchor
        lockLens = lens
        guard let a = anchor, let c = cameras else { return }
        if case .orbit = mode {
            orbit = OrbitCamera(eye: orbit.eye, target: a, up: c.up, fovy: Float(lens?.verticalFOV ?? Double(orbit.fovy)))
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
        let graded = look.flatMap { prepareGrade($0, width: drawable.texture.width, height: drawable.texture.height) }
        let drew = (try? renderer.render(viewports: [vp], colorTexture: graded?.source ?? drawable.texture,
                                         colorStoreAction: .store,
                                         depthTexture: view.depthStencilTexture, rasterizationRateMap: nil,
                                         renderTargetArrayLength: 0, to: cb)) ?? false
        if drew, let g = graded { encodeGrade(g, into: drawable.texture, cb: cb) }
        if drew { cb.present(drawable) }
        cb.commit()
    }
}

// MARK: - Move playback

/// One move loaded into a scene: which file, which frame, playing or not. Its own object so the
/// 30 fps frame counter re-renders only the views that show it, not every view of the scene.
@MainActor
final class MovePlayback: ObservableObject {
    /// "" = no move.
    @Published private(set) var path = ""
    @Published private(set) var move: MovePath?
    @Published private(set) var loadError: String?
    @Published var frame: Double = 0 { didSet { if !playing { show() } } }
    @Published private(set) var playing = false
    weak var scene: SplatScene?
    private var timer: Task<Void, Never>?

    var frameCount: Int { move?.frames.count ?? 0 }
    var label: String {
        if path.hasPrefix("keys:") { return memoryLabel }
        return path.isEmpty ? "" : MovePath.label(path)
    }
    var isKeyed: Bool { path.hasPrefix("keys:") }

    /// Load a move file. keepFrame: a recompile of the same move stays on the frame being
    /// looked at (clamped), so editing a late cue does not throw the view back to frame 1.
    func load(_ newPath: String, keepFrame: Bool = false) {
        pause()
        loadError = nil
        path = newPath
        guard !newPath.isEmpty else { move = nil; frame = 0; return }
        do {
            let m = try MovePath.load(path: newPath)
            move = m
            let f = keepFrame ? min(frame, Double(max(m.frames.count - 1, 0))) : 0
            if frame != f { frame = f } else { show() }
        } catch {
            move = nil
            loadError = error.localizedDescription
        }
    }

    /// A move that lives only in memory (the keyframe editor's bake): same player, no file.
    /// Stays on the frame being looked at, clamped, so an edit does not throw the view.
    func load(inMemory m: MovePath, label: String, show: Bool) {
        pause()
        loadError = nil
        path = "keys:" + label
        memoryLabel = label
        move = m
        let f = min(frame, Double(max(m.frames.count - 1, 0)))
        if frame != f { frame = f } else if show { self.show() }
    }
    private var memoryLabel = ""

    func clear() {
        pause()
        path = ""
        move = nil
        loadError = nil
        frame = 0
    }

    func toggle() { playing ? pause() : play() }

    func play() {
        guard let m = move, m.frames.count > 1 else { return }
        if Int(frame) >= m.frames.count - 1 { frame = 0 }
        playing = true
        show()
        let tick = UInt64(1_000_000_000 / max(m.fps, 1))
        timer?.cancel()
        timer = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: tick)
                guard let self = self, self.playing, let m = self.move, !m.frames.isEmpty else { return }
                self.frame = (self.frame + 1).truncatingRemainder(dividingBy: Double(m.frames.count))
                self.show()
            }
        }
    }

    func pause() {
        timer?.cancel()
        timer = nil
        if playing { playing = false }
    }

    /// Put the scene's camera on the current frame.
    func show() {
        guard let m = move, let p = m.pose(at: Int(frame)) else { return }
        scene?.stand(atMoveFrame: p, label: "\(label) · frame \(Int(frame) + 1) of \(m.frames.count)")
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
