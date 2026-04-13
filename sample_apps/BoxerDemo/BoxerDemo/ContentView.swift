import SwiftUI
import PhotosUI
import ARKit
import SceneKit
import UIKit

struct ContentView: View {
    @StateObject private var inference = BoxerInference()
    @StateObject private var arSession = ARSessionManager()
    @State private var mode: Mode = .ar
    @State private var detections: [Detection3D] = []
    @State private var isProcessing = false
    @State private var processingTime: Double?
    @State private var status = ""
    @State private var drawingBox = false
    @State private var boxStart: CGPoint = .zero
    @State private var boxEnd: CGPoint = .zero
    @State private var userBoxes: [CGRect] = []
    @State private var viewSize: CGSize = .zero
    @State private var selectedItem: PhotosPickerItem?
    @State private var inputImage: UIImage?
    @State private var sceneView: ARSCNView?

    enum Mode: String, CaseIterable { case ar = "AR Camera", photo = "Photo" }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                Picker("Mode", selection: $mode) {
                    ForEach(Mode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }.pickerStyle(.segmented).padding(.horizontal).padding(.top, 4)

                HStack {
                    Circle().fill(inference.isReady ? .green : .red).frame(width: 8, height: 8)
                    Text(inference.isReady
                         ? (mode == .ar ? (arSession.hasDepth ? "LiDAR + Pose" : "Pose only") : "Ready")
                         : "Loading…")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    if let t = processingTime {
                        Text(String(format: "%.2fs", t)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    if isProcessing { ProgressView().controlSize(.small) }
                }.padding(.horizontal).padding(.vertical, 4)

                GeometryReader { geo in
                    ZStack {
                        switch mode {
                        case .ar:
                            ARSceneViewContainer(session: arSession.session) { view in sceneView = view }
                        case .photo:
                            if let img = inputImage {
                                Image(uiImage: img).resizable().aspectRatio(contentMode: .fit)
                            } else {
                                VStack(spacing: 12) {
                                    Image(systemName: "cube.transparent").font(.system(size: 60)).foregroundStyle(.secondary)
                                    Text("Select a photo, draw boxes, lift to 3D").multilineTextAlignment(.center).foregroundStyle(.secondary)
                                }
                            }
                        }
                        // Box drawing overlay
                        drawingOverlay
                    }
                    .onAppear { viewSize = geo.size }
                    .onChange(of: geo.size) { _, s in viewSize = s }
                    .gesture(boxGesture)
                }

                if !detections.isEmpty { resultsBar }
                controlsBar
            }
            .navigationTitle("Boxer 3D")
            .navigationBarTitleDisplayMode(.inline)
        }
        .onChange(of: mode) { _, m in
            userBoxes.removeAll(); detections.removeAll(); clearSceneBoxes()
            if m == .ar { arSession.start() } else { arSession.stop() }
        }
        .onChange(of: selectedItem) { _, item in loadPhoto(item) }
        .onAppear { if mode == .ar { arSession.start() } }
        .onDisappear { arSession.stop() }
    }

    // MARK: - Drawing

    @ViewBuilder private var drawingOverlay: some View {
        ForEach(userBoxes.indices, id: \.self) { i in
            Rectangle().stroke(Color.cyan.opacity(0.7), lineWidth: 2)
                .background(Color.cyan.opacity(0.05))
                .frame(width: userBoxes[i].width, height: userBoxes[i].height)
                .position(x: userBoxes[i].midX, y: userBoxes[i].midY)
        }
        if drawingBox {
            let r = drawRect
            Rectangle().stroke(Color.yellow, lineWidth: 2).background(Color.yellow.opacity(0.05))
                .frame(width: r.width, height: r.height).position(x: r.midX, y: r.midY)
        }
        // 2D labels for each detection
        ForEach(detections) { det in
            let imgW: CGFloat = mode == .ar ? viewSize.width : CGFloat(inputImage?.cgImage?.width ?? 1)
            let imgH: CGFloat = mode == .ar ? viewSize.height : CGFloat(inputImage?.cgImage?.height ?? 1)
            VStack(spacing: 1) {
                Text(String(format: "%.1fm away", det.distance)).font(.system(size: 11, weight: .bold, design: .monospaced))
                Text(String(format: "%.2f×%.2f×%.2f m", det.size.x, det.size.y, det.size.z)).font(.system(size: 9, design: .monospaced))
                Text(String(format: "%.0f%%", det.confidence * 100)).font(.system(size: 9))
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(Color.cyan.opacity(0.85)).clipShape(RoundedRectangle(cornerRadius: 8))
            .position(x: det.box2D.midX * viewSize.width / imgW, y: det.box2D.midY * viewSize.height / imgH)
        }
    }

    private var drawRect: CGRect {
        CGRect(x: min(boxStart.x, boxEnd.x), y: min(boxStart.y, boxEnd.y),
               width: abs(boxEnd.x - boxStart.x), height: abs(boxEnd.y - boxStart.y))
    }
    private var boxGesture: some Gesture {
        DragGesture(minimumDistance: 10)
            .onChanged { v in if !drawingBox { boxStart = v.startLocation }; drawingBox = true; boxEnd = v.location }
            .onEnded { _ in drawingBox = false; let r = drawRect; if r.width > 20 && r.height > 20 { userBoxes.append(r) } }
    }

    // MARK: - Results

    private var resultsBar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(detections) { d in
                    Text(String(format: "%.2f×%.2f×%.2f m", d.size.x, d.size.y, d.size.z))
                        .font(.caption2.monospacedDigit())
                        .padding(.horizontal, 8).padding(.vertical, 4)
                        .background(.ultraThinMaterial).clipShape(Capsule())
                }
            }.padding(.horizontal)
        }.frame(height: 36)
    }

    private var controlsBar: some View {
        VStack(spacing: 8) {
            HStack(spacing: 12) {
                if mode == .photo {
                    PhotosPicker(selection: $selectedItem, matching: .images) {
                        Label("Photo", systemImage: "photo.badge.plus")
                    }.buttonStyle(.bordered)
                }
                if !userBoxes.isEmpty || !detections.isEmpty {
                    Button(role: .destructive) { userBoxes.removeAll(); detections.removeAll(); clearSceneBoxes() } label: {
                        Image(systemName: "trash")
                    }.buttonStyle(.bordered)
                }
                Button { Task { await runInference() } } label: {
                    Label("Lift to 3D", systemImage: "cube").frame(maxWidth: .infinity)
                }.buttonStyle(.borderedProminent)
                .disabled(!inference.isReady || userBoxes.isEmpty || isProcessing)
            }
        }.padding()
    }

    // MARK: - Actions

    private func loadPhoto(_ item: PhotosPickerItem?) {
        guard let item else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let img = UIImage(data: data) {
                await MainActor.run { inputImage = img; userBoxes.removeAll(); detections.removeAll() }
            }
        }
    }

    private func runInference() async {
        guard !userBoxes.isEmpty else { return }
        isProcessing = true; detections.removeAll(); clearSceneBoxes()
        let start = CFAbsoluteTimeGetCurrent()

        do {
            let results: [Detection3D]
            if mode == .ar {
                guard let frame = arSession.currentFrame else { isProcessing = false; return }
                let sx = CGFloat(frame.camera.imageResolution.width) / viewSize.width
                let sy = CGFloat(frame.camera.imageResolution.height) / viewSize.height
                let boxes: [[Float]] = userBoxes.map {
                    [Float($0.minX * sx), Float($0.minY * sy), Float($0.maxX * sx), Float($0.maxY * sy)]
                }
                results = try await inference.predict(frame: frame, boxes2D: boxes)
                // Place 3D boxes in AR scene
                addSceneBoxes(results)
            } else {
                guard let img = inputImage, let cg = img.cgImage else { isProcessing = false; return }
                let sx = CGFloat(cg.width) / viewSize.width, sy = CGFloat(cg.height) / viewSize.height
                let boxes: [[Float]] = userBoxes.map {
                    [Float($0.minX * sx), Float($0.minY * sy), Float($0.maxX * sx), Float($0.maxY * sy)]
                }
                results = try await inference.predict(image: img, boxes2D: boxes)
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            await MainActor.run { detections = results; processingTime = elapsed; isProcessing = false }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    // MARK: - SceneKit 3D Box Rendering

    private func clearSceneBoxes() {
        sceneView?.scene.rootNode.childNodes.filter { $0.name == "boxer3d" }.forEach { $0.removeFromParentNode() }
    }

    private func addSceneBoxes(_ dets: [Detection3D]) {
        guard let sceneView else { return }
        let colors: [UIColor] = [.systemCyan, .systemGreen, .systemOrange, .systemPink, .systemYellow]

        for (i, det) in dets.enumerated() {
            let color = colors[i % colors.count]
            let parent = SCNNode()
            parent.name = "boxer3d"
            parent.simdWorldTransform = det.worldTransform

            // Semi-transparent box fill
            let box = SCNBox(width: CGFloat(det.size.y), height: CGFloat(det.size.x),
                             length: CGFloat(det.size.z), chamferRadius: 0)
            let mat = SCNMaterial()
            mat.diffuse.contents = color.withAlphaComponent(0.2)
            mat.isDoubleSided = true
            box.firstMaterial = mat
            parent.addChildNode(SCNNode(geometry: box))

            // Wireframe edges (12 cylinders)
            let hw = det.size.y / 2, hh = det.size.x / 2, hd = det.size.z / 2
            let corners: [SIMD3<Float>] = [
                [-hw,-hh,-hd], [hw,-hh,-hd], [hw,-hh,hd], [-hw,-hh,hd],
                [-hw,hh,-hd],  [hw,hh,-hd],  [hw,hh,hd],  [-hw,hh,hd],
            ]
            let edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for (a, b) in edges {
                let edge = makeEdge(from: corners[a], to: corners[b], radius: 0.003, color: color)
                parent.addChildNode(edge)
            }

            // Label
            let text = SCNText(string: String(format: "%.0f×%.0f×%.0fcm", det.size.x*100, det.size.y*100, det.size.z*100), extrusionDepth: 0.5)
            text.font = UIFont.systemFont(ofSize: 3, weight: .bold)
            text.firstMaterial?.diffuse.contents = UIColor.white as Any
            let textNode = SCNNode(geometry: text)
            textNode.scale = SCNVector3(0.01, 0.01, 0.01)
            let (mn, mx) = textNode.boundingBox
            textNode.position = SCNVector3(-(mx.x - mn.x) * 0.005, Float(det.size.x / 2) + 0.02, 0)
            textNode.constraints = [SCNBillboardConstraint()]
            parent.addChildNode(textNode)

            sceneView.scene.rootNode.addChildNode(parent)
        }
    }

    private func makeEdge(from a: SIMD3<Float>, to b: SIMD3<Float>, radius: Float, color: UIColor) -> SCNNode {
        let diff = b - a
        let len = simd_length(diff)
        let cyl = SCNCylinder(radius: CGFloat(radius), height: CGFloat(len))
        cyl.firstMaterial?.diffuse.contents = color
        let node = SCNNode(geometry: cyl)
        node.simdPosition = (a + b) / 2
        let up = SIMD3<Float>(0, 1, 0)
        let dir = simd_normalize(diff)
        let dot = simd_dot(up, dir)
        if dot < -0.999 { node.simdOrientation = simd_quatf(angle: .pi, axis: SIMD3(1,0,0)) }
        else if dot < 0.999 { node.simdOrientation = simd_quatf(angle: acos(dot), axis: simd_normalize(simd_cross(up, dir))) }
        return node
    }
}

// MARK: - ARSCNView Container

struct ARSceneViewContainer: UIViewRepresentable {
    let session: ARSession
    var onCreated: ((ARSCNView) -> Void)?

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.session = session
        view.automaticallyUpdatesLighting = true
        onCreated?(view)
        return view
    }
    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}
