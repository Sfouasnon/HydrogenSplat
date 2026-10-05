import SwiftUI
import HSCore

/// Step 3 — Scan (optional). A LiDAR scan of the subject and the place, measured against the placed
/// cameras: it gives the project its real size and which way is up, a shape for training to start
/// from and a surface to hold the model to. The measuring and applying are `LidarScaleCard`
/// (`hs scale --lidar`); this page is where a person finds it, and it says what a scan is for and
/// what kind of file it takes (engine/hs/lidar.py `load_scan`: PLY, OBJ, text points, E57, LAS).
struct ScanStepPage: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary

    private var manifest: Manifest? { project.manifest }
    private var measured: Bool { manifest.flatMap { LidarCheck(manifest: $0) } != nil }

    var body: some View {
        StepPage {
            StepHeader(title: "Scan",
                       lead: "Optional. A LiDAR scan gives the model its real size, a shape to start from and a surface to hold to.")
            if let m = manifest {
                if !measured { whatToBring }
                LidarScaleCard(project: project, manifest: m)
                StepFooter(primary: "Go to Look",
                           note: "A project trains without a scan; the surface is then found from the photographs alone.") {
                    model.pipelineStage[project.path] = .look
                }
            } else {
                Text(project.manifestError ?? "This project has no manifest.").foregroundStyle(.red)
            }
        }
        .onAppear {
            // A scan that is applied should also be what training starts from and is held to. That
            // needs the init points written with it, which the settings leave off by default; a
            // project that has not touched these settings gets them on.
            if model.scaleSettings[project.path] == nil {
                var s = ScaleSettings()
                s.initPoints = true
                model.scaleSettings[project.path] = s
            }
        }
    }

    private var whatToBring: some View {
        VerdictCard(.info, headline: "What to bring",
                    detail: "A scan of the subject and what stands around it, saved as a point cloud or a mesh.") {
            VStack(alignment: .leading, spacing: 6) {
                Label("From a phone: PLY (point cloud or mesh), OBJ, or text points (XYZ, CSV, PTS). From a survey scanner: E57 or LAS. Not USDZ: export one of those instead.",
                      systemImage: "doc")
                Label("Units are read from the file, and which way is up from the scan; up can be set below if it is read wrong. A large scan is thinned as it is read.",
                      systemImage: "ruler")
                Label("Scan what the cameras saw, the ground and the backdrop included: the scan is matched to the points the cameras agree on.",
                      systemImage: "viewfinder")
            }
            .font(.callout)
            .fixedSize(horizontal: false, vertical: true)
        }
    }
}
