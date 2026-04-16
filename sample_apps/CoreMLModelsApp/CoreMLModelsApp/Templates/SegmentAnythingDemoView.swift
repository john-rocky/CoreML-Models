import SwiftUI
import PhotosUI
import CoreML
import UniformTypeIdentifiers

/// Segment Anything demo ported from SamKit.
/// Features: tap-to-segment, subject lifting (long press), glowing outline,
/// background dimming, lift context menu (copy/save/share), haptic feedback.
struct SegmentAnythingDemoView: View {
    let model: ModelEntry

    @State private var inputImage: UIImage?
    @State private var tapPoints: [CGPoint] = []  // in image coordinates
    @State private var isProcessing = false
    @State private var status = ""
    @State private var item: PhotosPickerItem?
    @State private var encoderModel: MLModel?
    @State private var decoderModel: MLModel?
    @State private var imageEmbedding: MLMultiArray?
    @State private var isEncoderLoaded = false

    // Mask state
    @State private var maskCGImage: CGImage?
    @State private var outlineImage: CGImage?

    // Lift state
    @State private var isLifted = false
    @State private var liftedImage: UIImage?
    @State private var liftDragOffset: CGSize = .zero
    @State private var liftStartTranslation: CGSize = .zero
    @State private var showLiftMenu = false
    @State private var toastMessage: String?
    @State private var showShareSheet = false

    // Gesture tracking
    @State private var gestureStartTime: Date?
    @State private var lastGestureTranslation: CGSize = .zero

    private var inputSize: Int { model.configInt("input_size") ?? 1024 }

    var body: some View {
        VStack(spacing: 0) {
            GeometryReader { geo in
                ZStack {
                    if let img = inputImage {
                        let fitted = fitSize(imageSize: img.size, in: geo.size)

                        // Base image
                        Image(uiImage: img)
                            .resizable().scaledToFit()
                            .frame(width: fitted.width, height: fitted.height)
                            .contentShape(Rectangle())
                            .gesture(
                                DragGesture(minimumDistance: 0)
                                    .onChanged { value in
                                        if gestureStartTime == nil {
                                            gestureStartTime = Date()
                                            // Long press to lift
                                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                                                guard gestureStartTime != nil, !isLifted, maskCGImage != nil else { return }
                                                let moved = hypot(lastGestureTranslation.width, lastGestureTranslation.height)
                                                guard moved < 15 else { return }
                                                liftStartTranslation = lastGestureTranslation
                                                handleLift()
                                            }
                                        }
                                        lastGestureTranslation = value.translation
                                        if isLifted {
                                            liftDragOffset = CGSize(
                                                width: value.translation.width - liftStartTranslation.width,
                                                height: value.translation.height - liftStartTranslation.height
                                            )
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
                                        // Short tap → add point
                                        if elapsed < 0.3 && moved < 15 && isEncoderLoaded {
                                            let imagePoint = viewToImageCoordinates(
                                                value.startLocation, viewSize: fitted, imageSize: img.size
                                            )
                                            tapPoints.append(imagePoint)
                                            runDecoder()
                                        }
                                    }
                            )

                        // Subject highlight: dim background + bright subject
                        if let mask = maskCGImage, !isLifted {
                            Color.black.opacity(0.25).allowsHitTesting(false)
                                .frame(width: fitted.width, height: fitted.height)

                            Image(uiImage: img)
                                .resizable().scaledToFit()
                                .frame(width: fitted.width, height: fitted.height)
                                .mask(
                                    Image(uiImage: UIImage(cgImage: mask))
                                        .resizable().scaledToFit()
                                        .frame(width: fitted.width, height: fitted.height)
                                )
                                .allowsHitTesting(false)
                        }

                        // Glowing outline
                        if !isLifted, let outline = outlineImage {
                            GlowingOutline(outline: outline, width: fitted.width, height: fitted.height)
                                .allowsHitTesting(false)
                        }

                        // Point markers
                        if !isLifted {
                            ForEach(tapPoints.indices, id: \.self) { i in
                                let viewPos = imageToViewCoordinates(
                                    tapPoints[i], viewSize: fitted, imageSize: img.size
                                )
                                Circle()
                                    .fill(.green)
                                    .frame(width: 14, height: 14)
                                    .overlay(Circle().stroke(.white, lineWidth: 2))
                                    .shadow(radius: 2)
                                    .position(viewPos)
                                    .allowsHitTesting(false)
                            }
                        }

                        // Lifted subject overlay
                        if isLifted {
                            Color.black.opacity(0.4)
                                .ignoresSafeArea()
                                .allowsHitTesting(showLiftMenu)
                                .contentShape(Rectangle())
                                .onTapGesture { dismissLift() }

                            if let mask = maskCGImage {
                                Image(uiImage: img)
                                    .resizable().scaledToFit()
                                    .frame(width: fitted.width, height: fitted.height)
                                    .mask(
                                        Image(uiImage: UIImage(cgImage: mask))
                                            .resizable().scaledToFit()
                                            .frame(width: fitted.width, height: fitted.height)
                                    )
                                    .shadow(color: .black.opacity(0.6), radius: 24, y: 12)
                                    .scaleEffect(showLiftMenu ? 1.0 : 1.05)
                                    .offset(liftDragOffset)
                                    .allowsHitTesting(false)
                                    .animation(.spring(response: 0.35, dampingFraction: 0.75), value: showLiftMenu)
                            }

                            if showLiftMenu {
                                liftContextMenu
                                    .transition(.scale(scale: 0.8).combined(with: .opacity))
                            }
                        }

                    } else {
                        VStack(spacing: 12) {
                            Image(systemName: "hand.tap")
                                .font(.system(size: 60)).foregroundStyle(.secondary)
                            Text("Select a photo, then tap to segment")
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }

                    // Processing
                    if isProcessing {
                        Color.black.opacity(0.3).allowsHitTesting(false)
                        ProgressView().tint(.white).scaleEffect(1.5)
                    }
                }
            }

            // Controls
            VStack(spacing: 8) {
                HStack {
                    if !isEncoderLoaded && inputImage != nil && !isProcessing {
                        Label("Encoding image…", systemImage: "brain")
                            .font(.caption).foregroundStyle(.orange)
                    }
                    Spacer()
                    if isProcessing {
                        Text(status).font(.caption).foregroundStyle(.secondary)
                    }
                }

                HStack(spacing: 12) {
                    PhotosPicker(selection: $item, matching: .images) {
                        Label("Photo", systemImage: "photo.badge.plus")
                    }
                    .buttonStyle(.bordered)
                    .disabled(isProcessing)

                    if !tapPoints.isEmpty {
                        Button {
                            tapPoints.removeAll()
                            maskCGImage = nil; outlineImage = nil; liftedImage = nil
                        } label: {
                            Label("Clear", systemImage: "arrow.counterclockwise")
                        }
                        .buttonStyle(.bordered)
                    }

                    Spacer()

                    if let output = liftedImage {
                        Button {
                            UIImageWriteToSavedPhotosAlbum(output, nil, nil, nil)
                            showToast("Saved to Photos")
                        } label: {
                            Image(systemName: "arrow.down.to.line")
                        }
                        .buttonStyle(.bordered)
                    }
                }
            }
            .padding()
        }
        .overlay { toastOverlay }
        .sheet(isPresented: $showShareSheet) {
            if let lifted = liftedImage {
                ActivityView(items: [lifted])
            }
        }
        .onChange(of: item) { _, _ in loadPhoto() }
        .animation(.easeInOut(duration: 0.25), value: isLifted)
        .animation(.spring(response: 0.3, dampingFraction: 0.75), value: showLiftMenu)
    }

    // MARK: - Lift context menu

    @ViewBuilder
    private var liftContextMenu: some View {
        VStack(spacing: 0) {
            Button {
                if let img = liftedImage, let data = img.pngData() {
                    UIPasteboard.general.setData(data, forPasteboardType: UTType.png.identifier)
                    dismissLift(); showToast("Copied to clipboard")
                }
            } label: {
                Label("Copy", systemImage: "doc.on.doc")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16).padding(.vertical, 12)
            }
            Divider()
            Button {
                if let img = liftedImage {
                    UIImageWriteToSavedPhotosAlbum(img, nil, nil, nil)
                    dismissLift(); showToast("Saved to Photos")
                }
            } label: {
                Label("Save to Photos", systemImage: "square.and.arrow.down")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16).padding(.vertical, 12)
            }
            Divider()
            Button {
                showShareSheet = true; dismissLift()
            } label: {
                Label("Share...", systemImage: "square.and.arrow.up")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16).padding(.vertical, 12)
            }
        }
        .foregroundColor(.primary).font(.body)
        .frame(width: 220)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .shadow(color: .black.opacity(0.2), radius: 20, y: 5)
    }

    // MARK: - Toast

    @ViewBuilder
    private var toastOverlay: some View {
        if let msg = toastMessage {
            VStack {
                Spacer()
                Text(msg)
                    .font(.subheadline.weight(.medium))
                    .foregroundColor(.white)
                    .padding(.horizontal, 16).padding(.vertical, 10)
                    .background(Capsule().fill(Color.black.opacity(0.75)))
                    .padding(.bottom, 60)
            }
            .transition(.move(edge: .bottom).combined(with: .opacity))
        }
    }

    private func showToast(_ msg: String) {
        withAnimation { toastMessage = msg }
        Task {
            try? await Task.sleep(for: .seconds(1.5))
            await MainActor.run { withAnimation { toastMessage = nil } }
        }
    }

    // MARK: - Lift

    private func handleLift() {
        guard let inputImage, let mask = maskCGImage,
              let cgImage = ImageUtils.normalizeOrientation(inputImage) else { return }

        // Extract masked object as PNG with transparency
        let w = cgImage.width, h = cgImage.height
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
        // Draw original
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let imageData = ctx.data else { return }
        let pixels = imageData.bindMemory(to: UInt8.self, capacity: w * h * 4)

        // Apply mask as alpha
        guard let maskCtx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                      bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
        maskCtx.draw(mask, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let maskData = maskCtx.data else { return }
        let maskPixels = maskData.bindMemory(to: UInt8.self, capacity: w * h * 4)

        for i in 0..<(w * h) {
            let alpha = maskPixels[i * 4 + 3]
            if alpha < 128 {
                pixels[i * 4] = 0; pixels[i * 4 + 1] = 0
                pixels[i * 4 + 2] = 0; pixels[i * 4 + 3] = 0
            }
        }

        guard let extracted = ctx.makeImage() else { return }
        liftedImage = UIImage(cgImage: extracted)

        let generator = UIImpactFeedbackGenerator(style: .medium)
        generator.impactOccurred()

        withAnimation(.spring(response: 0.35, dampingFraction: 0.75)) {
            isLifted = true
        }
    }

    private func dismissLift() {
        withAnimation(.easeOut(duration: 0.2)) {
            showLiftMenu = false; isLifted = false
        }
        liftDragOffset = .zero; liftStartTranslation = .zero
    }

    // MARK: - Photo loading & encoding

    private func loadPhoto() {
        guard let item else { return }
        tapPoints.removeAll(); maskCGImage = nil; outlineImage = nil
        liftedImage = nil; imageEmbedding = nil; isEncoderLoaded = false
        isProcessing = true; status = "Loading…"

        Task {
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let img = UIImage(data: data) else {
                await MainActor.run { isProcessing = false; status = "Failed" }; return
            }
            await MainActor.run { inputImage = img; status = "Loading encoder…" }
            await encodeImage(img)
        }
    }

    private func encodeImage(_ image: UIImage) async {
        do {
            let encName = model.configString("encoder") ?? model.files.first?.name ?? ""
            let enc = try await ModelLoader.load(for: model, named: encName)
            let decName = model.configString("decoder")
                ?? model.files.dropFirst().first?.name ?? model.files.last?.name ?? ""
            let dec = try await ModelLoader.load(for: model, named: decName)

            await MainActor.run { status = "Encoding image…" }
            guard let cgImage = ImageUtils.normalizeOrientation(image),
                  let pb = ImageUtils.pixelBuffer(from: cgImage, width: inputSize, height: inputSize) else {
                await MainActor.run { isProcessing = false; status = "Image error" }; return
            }

            let inputName = enc.modelDescription.inputDescriptionsByName.keys.first ?? "image"
            let input = try MLDictionaryFeatureProvider(dictionary: [inputName: pb])
            let encOutput = try await enc.prediction(from: input)

            var embedding: MLMultiArray?
            for name in encOutput.featureNames {
                if let arr = encOutput.featureValue(for: name)?.multiArrayValue {
                    embedding = arr; break
                }
            }

            guard let emb = embedding else {
                await MainActor.run { isProcessing = false; status = "Encoding failed" }; return
            }

            await MainActor.run {
                encoderModel = enc; decoderModel = dec; imageEmbedding = emb
                isEncoderLoaded = true; isProcessing = false; status = ""
            }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    // MARK: - Decode mask

    private func runDecoder() {
        guard let decoderModel, let imageEmbedding, let inputImage,
              !tapPoints.isEmpty else { return }
        isProcessing = true; status = "Segmenting…"

        Task {
            do {
                let n = tapPoints.count
                let coords = try MLMultiArray(shape: [1, NSNumber(value: n), 2], dataType: .float32)
                let labels = try MLMultiArray(shape: [1, NSNumber(value: n)], dataType: .float32)
                let ptr = coords.dataPointer.assumingMemoryBound(to: Float.self)
                let lptr = labels.dataPointer.assumingMemoryBound(to: Float.self)
                for i in 0..<n {
                    ptr[i * 2] = Float(tapPoints[i].x) / Float(inputImage.size.width) * Float(inputSize)
                    ptr[i * 2 + 1] = Float(tapPoints[i].y) / Float(inputImage.size.height) * Float(inputSize)
                    lptr[i] = 1.0
                }

                let desc = decoderModel.modelDescription.inputDescriptionsByName
                var inputDict: [String: MLFeatureValue] = [:]
                for (name, fd) in desc {
                    if fd.type == .multiArray {
                        let shape = fd.multiArrayConstraint?.shape.map { $0.intValue } ?? []
                        if shape.last == 2 || name.lowercased().contains("point") || name.lowercased().contains("coord") {
                            inputDict[name] = MLFeatureValue(multiArray: coords)
                        } else if name.lowercased().contains("label") {
                            inputDict[name] = MLFeatureValue(multiArray: labels)
                        } else {
                            inputDict[name] = MLFeatureValue(multiArray: imageEmbedding)
                        }
                    }
                }

                let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
                let output = try await decoderModel.prediction(from: input)

                var maskArr: MLMultiArray?
                for name in output.featureNames {
                    if let arr = output.featureValue(for: name)?.multiArrayValue {
                        maskArr = arr; break
                    }
                }

                guard let mask = maskArr else {
                    await MainActor.run { isProcessing = false; status = "No mask" }; return
                }

                // Build binary mask CGImage
                let binaryMask = buildBinaryMask(from: mask, width: Int(inputImage.size.width),
                                                  height: Int(inputImage.size.height))
                let outline = binaryMask.flatMap { generateOutline(from: $0) }

                await MainActor.run {
                    maskCGImage = binaryMask
                    outlineImage = outline
                    isProcessing = false; status = ""
                }
            } catch {
                await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
            }
        }
    }

    // MARK: - Mask processing

    private func buildBinaryMask(from arr: MLMultiArray, width: Int, height: Int) -> CGImage? {
        let shape = arr.shape.map { $0.intValue }
        let mH = shape[shape.count - 2]
        let mW = shape[shape.count - 1]
        let total = arr.count
        let offset = total - mH * mW

        guard let ctx = CGContext(data: nil, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: width * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        guard let data = ctx.data else { return nil }
        let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)

        for y in 0..<height {
            let my = min(y * mH / height, mH - 1)
            for x in 0..<width {
                let mx = min(x * mW / width, mW - 1)
                let val = ImageUtils.readFloat(arr, at: offset + my * mW + mx)
                let o = (y * width + x) * 4
                if val > 0 {
                    pixels[o] = 255; pixels[o + 1] = 255; pixels[o + 2] = 255; pixels[o + 3] = 255
                } else {
                    pixels[o] = 0; pixels[o + 1] = 0; pixels[o + 2] = 0; pixels[o + 3] = 0
                }
            }
        }
        return ctx.makeImage()
    }

    private func generateOutline(from mask: CGImage) -> CGImage? {
        let w = mask.width, h = mask.height
        let rect = CGRect(x: 0, y: 0, width: w, height: h)
        let glowRadius = CGFloat(min(30, max(4, min(w, h) / 100)))

        // White silhouette from mask
        guard let silCtx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                     bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        silCtx.draw(mask, in: rect)
        silCtx.setBlendMode(.sourceIn)
        silCtx.setFillColor(UIColor.white.cgColor)
        silCtx.fill(rect)
        guard let whiteSil = silCtx.makeImage() else { return nil }

        // Glow outline = shadow of silhouette minus the silhouette itself
        guard let outCtx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                     bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        outCtx.setShadow(offset: .zero, blur: glowRadius, color: UIColor.white.cgColor)
        outCtx.draw(whiteSil, in: rect)
        outCtx.setShadow(offset: .zero, blur: 0, color: nil)
        outCtx.setBlendMode(.destinationOut)
        outCtx.draw(whiteSil, in: rect)
        return outCtx.makeImage()
    }

    // MARK: - Coordinate helpers

    private func viewToImageCoordinates(_ point: CGPoint, viewSize: CGSize, imageSize: CGSize) -> CGPoint {
        CGPoint(x: point.x / viewSize.width * imageSize.width,
                y: point.y / viewSize.height * imageSize.height)
    }

    private func imageToViewCoordinates(_ point: CGPoint, viewSize: CGSize, imageSize: CGSize) -> CGPoint {
        CGPoint(x: point.x / imageSize.width * viewSize.width,
                y: point.y / imageSize.height * viewSize.height)
    }

    private func fitSize(imageSize: CGSize, in containerSize: CGSize) -> CGSize {
        let scale = min(containerSize.width / imageSize.width, containerSize.height / imageSize.height)
        return CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
    }
}

// MARK: - Glowing outline animation

private struct GlowingOutline: View {
    let outline: CGImage
    let width: CGFloat
    let height: CGFloat

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30)) { timeline in
            let t = timeline.date.timeIntervalSinceReferenceDate
            let phase = t.truncatingRemainder(dividingBy: 2.5) / 2.5

            ZStack {
                Image(uiImage: UIImage(cgImage: outline))
                    .resizable().scaledToFit()
                    .frame(width: width, height: height)
                    .colorMultiply(Color(red: 0.5, green: 0.85, blue: 1.0))
                    .blur(radius: 5).opacity(0.8)

                Image(uiImage: UIImage(cgImage: outline))
                    .resizable().scaledToFit()
                    .frame(width: width, height: height)
                    .colorMultiply(.white)

                Image(uiImage: UIImage(cgImage: outline))
                    .resizable().scaledToFit()
                    .frame(width: width, height: height)
                    .colorMultiply(.white)
                    .mask(
                        AngularGradient(
                            gradient: Gradient(colors: [
                                .white, .white.opacity(0.5), .clear, .clear,
                                .clear, .clear, .clear, .white.opacity(0.3)
                            ]),
                            center: .center,
                            startAngle: .degrees(phase * 360),
                            endAngle: .degrees(phase * 360 + 360)
                        )
                    )
                    .blur(radius: 2)
            }
        }
    }
}

// MARK: - Share sheet

private struct ActivityView: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
