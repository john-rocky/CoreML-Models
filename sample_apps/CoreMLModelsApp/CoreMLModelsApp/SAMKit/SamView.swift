import SwiftUI
import UIKit
import UniformTypeIdentifiers
import Photos

/// Vendored from SAMKitUI with the point + negative-point + box tool modes
/// from `UnifiedSegmentationView.swift` (samples/ios-sample/SAMKitDemo/)
/// merged back in. Text detection and SAM2 support are intentionally omitted —
/// hub-app MobileSAM only needs point + box.

enum ToolMode {
    case point
    case box
}

struct SamView: View {
    let image: UIImage
    let model: SamModelRef
    let config: RuntimeConfig

    @State private var session: SamSession?
    @State private var loadError: Error?

    init(image: UIImage, model: SamModelRef, config: RuntimeConfig = .bestAvailable) {
        self.image = image
        self.model = model
        self.config = config
    }

    var body: some View {
        VStack {
            if let error = loadError {
                ErrorView(error: error)
            } else if let session {
                InteractiveSegmentationView(image: image, session: session)
            } else {
                LoadingView(modelType: model.modelType)
            }
        }
        .onAppear { loadSession() }
    }

    private func loadSession() {
        Task {
            do {
                let newSession = try SamSession(model: model, config: config)
                await MainActor.run { self.session = newSession }
            } catch {
                await MainActor.run { self.loadError = error }
            }
        }
    }
}

struct InteractiveSegmentationView: View {
    let image: UIImage
    let session: SamSession

    // Prompt state
    @State private var points: [SamPoint] = []
    @State private var boundingBox: SamBox?
    @State private var result: SamResult?
    @State private var isProcessing = false
    @State private var processingMessage: String = ""
    @State private var selectedMaskIndex = 0
    @State private var errorMessage: String?
    @State private var samImageSet = false

    // Tool selection
    @State private var toolMode: ToolMode = .point
    @State private var showNegativePoints = false
    @State private var dragStart: CGPoint?
    @State private var dragEnd: CGPoint?

    // Visuals derived from the selected mask
    @State private var binaryMask: CGImage?
    @State private var outlineImage: CGImage?

    // Subject-lift state
    @State private var liftedImage: UIImage?
    @State private var isLifted = false
    @State private var liftDragOffset: CGSize = .zero
    @State private var liftStartTranslation: CGSize = .zero
    @State private var showLiftMenu = false
    @State private var toastMessage: String?
    @State private var showShareSheet = false

    // Unified gesture tracking
    @State private var gestureStartTime: Date?
    @State private var lastGestureTranslation: CGSize = .zero

    private var hasVisibleMasks: Bool {
        result?.masks.isEmpty == false
    }

    var body: some View {
        VStack(spacing: 0) {
            GeometryReader { geometry in
                ZStack {
                    // Base image — single layout, all hit-testing relative to this.
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(width: geometry.size.width, height: geometry.size.height)
                        .contentShape(Rectangle())
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { value in
                                    if gestureStartTime == nil {
                                        gestureStartTime = Date()
                                        // Long-press-to-lift — fires only if we
                                        // already have a mask and the finger
                                        // has barely moved.
                                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                                            guard gestureStartTime != nil,
                                                  !isLifted,
                                                  hasVisibleMasks,
                                                  dragStart == nil else { return }
                                            let moved = hypot(lastGestureTranslation.width, lastGestureTranslation.height)
                                            guard moved < 15 else { return }
                                            liftStartTranslation = lastGestureTranslation
                                            handleLiftObject()
                                        }
                                    }
                                    lastGestureTranslation = value.translation

                                    if isLifted {
                                        liftDragOffset = CGSize(
                                            width: value.translation.width - liftStartTranslation.width,
                                            height: value.translation.height - liftStartTranslation.height
                                        )
                                        return
                                    }

                                    let moved = hypot(value.translation.width, value.translation.height)
                                    if toolMode == .box && moved >= 10 {
                                        if dragStart == nil { dragStart = value.startLocation }
                                        dragEnd = value.location
                                    }
                                }
                                .onEnded { value in
                                    let elapsed = Date().timeIntervalSince(gestureStartTime ?? Date())
                                    let moved = hypot(value.translation.width, value.translation.height)
                                    gestureStartTime = nil
                                    lastGestureTranslation = .zero

                                    if isLifted {
                                        withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                                            liftDragOffset = .zero
                                        }
                                        withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                                            showLiftMenu = true
                                        }
                                        return
                                    }

                                    // Box drag finished → set bounding box, run segmentation.
                                    if toolMode == .box, let start = dragStart {
                                        let startPt = viewToImage(start, viewSize: geometry.size)
                                        let endPt = viewToImage(value.location, viewSize: geometry.size)
                                        boundingBox = SamBox(
                                            x0: Float(min(startPt.x, endPt.x)),
                                            y0: Float(min(startPt.y, endPt.y)),
                                            x1: Float(max(startPt.x, endPt.x)),
                                            y1: Float(max(startPt.y, endPt.y))
                                        )
                                        dragStart = nil
                                        dragEnd = nil
                                        runSegmentation()
                                        return
                                    }

                                    // Short tap → add point.
                                    if elapsed < 0.3 && moved < 15 {
                                        let imagePoint = viewToImage(value.startLocation, viewSize: geometry.size)
                                        addPoint(at: imagePoint)
                                    }
                                }
                        )

                    // Subject highlight (dim + bright subject)
                    if let mask = binaryMask, !isLifted {
                        Color.black.opacity(0.25).allowsHitTesting(false)

                        Image(uiImage: image)
                            .resizable().scaledToFit()
                            .frame(width: geometry.size.width, height: geometry.size.height)
                            .mask(
                                Image(uiImage: UIImage(cgImage: mask))
                                    .resizable().scaledToFit()
                                    .frame(width: geometry.size.width, height: geometry.size.height)
                            )
                            .allowsHitTesting(false)
                    }

                    // Glowing outline
                    if !isLifted, let outline = outlineImage {
                        GlowingOutlineView(outline: outline,
                                           width: geometry.size.width,
                                           height: geometry.size.height)
                    }

                    // Drag-in-progress box
                    if let start = dragStart, let end = dragEnd {
                        Rectangle()
                            .stroke(Color.blue, lineWidth: 2)
                            .background(Color.blue.opacity(0.1))
                            .frame(width: abs(end.x - start.x), height: abs(end.y - start.y))
                            .position(
                                x: min(start.x, end.x) + abs(end.x - start.x) / 2,
                                y: min(start.y, end.y) + abs(end.y - start.y) / 2
                            )
                            .allowsHitTesting(false)
                    }

                    // Committed bounding box
                    if let box = boundingBox {
                        let topLeft = imageToView(
                            CGPoint(x: CGFloat(box.x0), y: CGFloat(box.y0)),
                            viewSize: geometry.size
                        )
                        let bottomRight = imageToView(
                            CGPoint(x: CGFloat(box.x1), y: CGFloat(box.y1)),
                            viewSize: geometry.size
                        )
                        Rectangle()
                            .stroke(Color.blue, lineWidth: 2)
                            .frame(
                                width: abs(bottomRight.x - topLeft.x),
                                height: abs(bottomRight.y - topLeft.y)
                            )
                            .position(
                                x: topLeft.x + (bottomRight.x - topLeft.x) / 2,
                                y: topLeft.y + (bottomRight.y - topLeft.y) / 2
                            )
                            .allowsHitTesting(false)
                    }

                    // Point markers
                    if !isLifted {
                        ForEach(Array(points.enumerated()), id: \.offset) { _, point in
                            let pos = imageToView(
                                CGPoint(x: point.x, y: point.y), viewSize: geometry.size
                            )
                            Circle()
                                .fill(point.label == .positive ? Color.green : Color.red)
                                .frame(width: 12, height: 12)
                                .overlay(Circle().stroke(Color.white, lineWidth: 2))
                                .position(pos)
                                .allowsHitTesting(false)
                        }
                    }

                    // Processing indicator
                    if isProcessing {
                        Color.black.opacity(0.3).allowsHitTesting(false)
                        VStack(spacing: 12) {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .scaleEffect(1.5)
                            if !processingMessage.isEmpty {
                                Text(processingMessage)
                                    .font(.subheadline.weight(.medium))
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 14)
                                    .padding(.vertical, 8)
                                    .background(
                                        RoundedRectangle(cornerRadius: 10)
                                            .fill(Color.black.opacity(0.55))
                                    )
                            }
                        }
                        .allowsHitTesting(false)
                    }

                    // Floating toolbar (point / negative point / box / clear)
                    if !isLifted {
                        VStack {
                            floatingToolbar
                            Spacer()
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 8)
                        .padding(.leading, 8)
                    }

                    // Error banner
                    if let error = errorMessage {
                        VStack {
                            Text(error)
                                .font(.caption)
                                .foregroundColor(.white)
                                .padding(.horizontal, 12).padding(.vertical, 6)
                                .background(RoundedRectangle(cornerRadius: 8).fill(Color.red.opacity(0.8)))
                                .padding(.top, 56)
                            Spacer()
                        }
                        .frame(maxWidth: .infinity)
                        .allowsHitTesting(false)
                    }

                    // Lift overlay
                    if isLifted {
                        Color.black.opacity(0.4)
                            .ignoresSafeArea()
                            .allowsHitTesting(showLiftMenu)
                            .contentShape(Rectangle())
                            .onTapGesture { dismissLift() }

                        if let mask = binaryMask {
                            Image(uiImage: image)
                                .resizable().scaledToFit()
                                .frame(width: geometry.size.width, height: geometry.size.height)
                                .mask(
                                    Image(uiImage: UIImage(cgImage: mask))
                                        .resizable().scaledToFit()
                                        .frame(width: geometry.size.width, height: geometry.size.height)
                                )
                                .shadow(color: .black.opacity(0.6), radius: 24, y: 12)
                                .scaleEffect(showLiftMenu ? 1.0 : 1.05)
                                .offset(liftDragOffset)
                                .allowsHitTesting(false)
                                .animation(.spring(response: 0.35, dampingFraction: 0.75),
                                           value: showLiftMenu)
                        }

                        if showLiftMenu {
                            LiftContextMenuView(
                                onCopy: { performCopy() },
                                onSave: { performSave() },
                                onShare: { showShareSheet = true; dismissLift() }
                            )
                            .transition(.scale(scale: 0.8).combined(with: .opacity))
                        }
                    }
                }
            }

            // Mask picker (shown when multiple candidate masks are returned)
            if let result, result.masks.count > 1 {
                VStack(spacing: 6) {
                    Text("Masks (\(result.masks.count))")
                        .font(.caption)
                    Picker("Mask", selection: $selectedMaskIndex) {
                        ForEach(0..<result.masks.count, id: \.self) { idx in
                            Text("Mask \(idx + 1) (\(String(format: "%.3f", result.scores[idx])))")
                                .tag(idx)
                        }
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: selectedMaskIndex) { _, _ in updateMaskAndOutline() }
                }
                .padding(.horizontal, 12)
                .padding(.bottom, 8)
            }
        }
        .overlay { ToastOverlay(message: toastMessage) }
        .sheet(isPresented: $showShareSheet) {
            if let lifted = liftedImage { ActivityViewController(items: [lifted]) }
        }
        .animation(.easeInOut(duration: 0.25), value: isLifted)
        .animation(.easeInOut(duration: 0.2), value: toastMessage != nil)
        .animation(.spring(response: 0.3, dampingFraction: 0.75), value: showLiftMenu)
    }

    // MARK: - Floating toolbar

    @ViewBuilder
    private var floatingToolbar: some View {
        HStack(spacing: 6) {
            Button {
                toolMode = .point
                showNegativePoints = false
            } label: {
                Image(systemName: "hand.point.up.left")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(toolMode == .point && !showNegativePoints ? .white : .primary)
                    .frame(width: 36, height: 36)
                    .background(toolMode == .point && !showNegativePoints ? Color.blue : Color.clear)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            Button {
                toolMode = .point
                showNegativePoints = true
            } label: {
                Image(systemName: "minus.circle")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(toolMode == .point && showNegativePoints ? .white : .primary)
                    .frame(width: 36, height: 36)
                    .background(toolMode == .point && showNegativePoints ? Color.red : Color.clear)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            Button {
                toolMode = .box
            } label: {
                Image(systemName: "rectangle.dashed")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(toolMode == .box ? .white : .primary)
                    .frame(width: 36, height: 36)
                    .background(toolMode == .box ? Color.blue : Color.clear)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            Divider().frame(height: 24)

            Button {
                clearAll()
            } label: {
                Image(systemName: "trash")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(.red)
                    .frame(width: 36, height: 36)
            }
            .disabled(points.isEmpty && boundingBox == nil && result == nil)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .shadow(color: .black.opacity(0.15), radius: 4, y: 2)
    }

    // MARK: - Prompt actions

    private func addPoint(at location: CGPoint) {
        guard toolMode == .point else { return }
        if points.count >= PromptEncoder.maxPoints {
            points.removeFirst()
        }
        let label: SamPointLabel = showNegativePoints ? .negative : .positive
        points.append(SamPoint(x: location.x, y: location.y, label: label))
        runSegmentation()
    }

    private func clearAll() {
        points.removeAll()
        boundingBox = nil
        result = nil
        binaryMask = nil
        outlineImage = nil
        errorMessage = nil
        selectedMaskIndex = 0
    }

    private func runSegmentation() {
        guard !points.isEmpty || boundingBox != nil else { return }
        isProcessing = true
        processingMessage = samImageSet ? "Segmenting…" : "Encoding image…"
        errorMessage = nil

        Task {
            do {
                if !samImageSet, let cg = orientedCGImage(from: image) {
                    try session.setImage(cg)
                    await MainActor.run {
                        samImageSet = true
                        processingMessage = "Segmenting…"
                    }
                }
                let newResult = try session.predict(points: points, box: boundingBox)
                await MainActor.run {
                    self.result = newResult
                    self.selectedMaskIndex = 0
                    self.isProcessing = false
                    self.processingMessage = ""
                }
                updateMaskAndOutline()
            } catch {
                await MainActor.run {
                    self.isProcessing = false
                    self.processingMessage = ""
                    self.errorMessage = error.localizedDescription
                }
            }
        }
    }

    /// Redraws the UIImage so the returned CGImage has the bitmap orientation
    /// baked in. `UIImage.cgImage` otherwise returns the raw sensor bitmap and
    /// ignores `imageOrientation`, which causes camera photos (typically
    /// `.right`) to feed the encoder rotated 90°.
    private func orientedCGImage(from image: UIImage) -> CGImage? {
        if image.imageOrientation == .up, let cg = image.cgImage { return cg }
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: image.size, format: format)
        let redrawn = renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: image.size))
        }
        return redrawn.cgImage
    }

    // MARK: - Mask visuals

    private var currentMaskCG: CGImage? {
        guard let r = result, r.masks.indices.contains(selectedMaskIndex) else { return nil }
        return r.masks[selectedMaskIndex].cgImage
    }

    private func updateMaskAndOutline() {
        guard let mask = currentMaskCG else {
            binaryMask = nil
            outlineImage = nil
            return
        }
        Task {
            if let processed = processVisibleMasks([mask]) {
                await MainActor.run {
                    binaryMask = processed.binary
                    outlineImage = processed.outline
                }
            }
        }
    }

    // MARK: - Subject lift

    private func handleLiftObject() {
        guard let cgImage = orientedCGImage(from: image),
              let r = result, r.masks.indices.contains(selectedMaskIndex) else { return }
        let mask = r.masks[selectedMaskIndex]

        guard let extracted = SamMask.extractObject(from: cgImage, masks: [mask]) else { return }
        liftedImage = UIImage(cgImage: extracted)

        UIImpactFeedbackGenerator(style: .medium).impactOccurred()

        withAnimation(.spring(response: 0.35, dampingFraction: 0.75)) {
            isLifted = true
        }
    }

    private func dismissLift() {
        withAnimation(.easeOut(duration: 0.2)) {
            showLiftMenu = false
            isLifted = false
        }
        liftDragOffset = .zero
        liftStartTranslation = .zero
    }

    private func performCopy() {
        guard let lifted = liftedImage else { return }
        let msg = copyObject(lifted)
        dismissLift()
        withAnimation { toastMessage = msg }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            withAnimation { toastMessage = nil }
        }
    }

    private func performSave() {
        guard let lifted = liftedImage else { return }
        dismissLift()
        saveObject(lifted) { msg in
            withAnimation { toastMessage = msg }
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                withAnimation { toastMessage = nil }
            }
        }
    }

    // MARK: - View ↔ image coordinate mapping (fixed letterbox)

    private func viewToImage(_ viewPoint: CGPoint, viewSize: CGSize) -> CGPoint {
        let imageSize = image.size
        let imageAspect = imageSize.width / imageSize.height
        let viewAspect = viewSize.width / viewSize.height

        let displayed: CGSize
        let offset: CGPoint
        if imageAspect > viewAspect {
            let w = viewSize.width
            let h = w / imageAspect
            displayed = CGSize(width: w, height: h)
            offset = CGPoint(x: 0, y: (viewSize.height - h) / 2)
        } else {
            let h = viewSize.height
            let w = h * imageAspect
            displayed = CGSize(width: w, height: h)
            offset = CGPoint(x: (viewSize.width - w) / 2, y: 0)
        }

        let x = (viewPoint.x - offset.x) / displayed.width * imageSize.width
        let y = (viewPoint.y - offset.y) / displayed.height * imageSize.height
        return CGPoint(x: min(max(x, 0), imageSize.width),
                       y: min(max(y, 0), imageSize.height))
    }

    private func imageToView(_ imagePoint: CGPoint, viewSize: CGSize) -> CGPoint {
        let imageSize = image.size
        let imageAspect = imageSize.width / imageSize.height
        let viewAspect = viewSize.width / viewSize.height

        let displayed: CGSize
        let offset: CGPoint
        if imageAspect > viewAspect {
            let w = viewSize.width
            let h = w / imageAspect
            displayed = CGSize(width: w, height: h)
            offset = CGPoint(x: 0, y: (viewSize.height - h) / 2)
        } else {
            let h = viewSize.height
            let w = h * imageAspect
            displayed = CGSize(width: w, height: h)
            offset = CGPoint(x: (viewSize.width - w) / 2, y: 0)
        }

        return CGPoint(
            x: imagePoint.x / imageSize.width * displayed.width + offset.x,
            y: imagePoint.y / imageSize.height * displayed.height + offset.y
        )
    }
}

struct LoadingView: View {
    let modelType: ModelType

    var body: some View {
        VStack {
            ProgressView()
            Text("Loading \(modelType.modelName)...")
                .padding(.top)
        }
    }
}

struct ErrorView: View {
    let error: Error

    var body: some View {
        VStack {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 50))
                .foregroundColor(.red)
            Text("Error")
                .font(.headline)
                .padding(.top)
            Text(error.localizedDescription)
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding()
        }
    }
}
