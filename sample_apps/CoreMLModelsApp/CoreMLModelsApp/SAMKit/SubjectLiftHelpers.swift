import SwiftUI
import UIKit
import UniformTypeIdentifiers
import Photos

// Vendored verbatim from SAMKitUI/SubjectLiftHelpers.swift.
// Same-target so `public` modifiers are dropped.

// MARK: - Mask Processing

func binarizeMask(_ maskImage: CGImage) -> CGImage? {
    let width = maskImage.width
    let height = maskImage.height

    guard let ctx = CGContext(
        data: nil, width: width, height: height,
        bitsPerComponent: 8, bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }

    let rect = CGRect(x: 0, y: 0, width: width, height: height)
    ctx.draw(maskImage, in: rect)

    guard let data = ctx.data else { return nil }
    let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)

    let threshold: UInt8 = 128
    for i in 0..<(width * height) {
        let o = i * 4
        if pixels[o + 3] >= threshold {
            pixels[o] = 255; pixels[o + 1] = 255; pixels[o + 2] = 255; pixels[o + 3] = 255
        } else {
            pixels[o] = 0; pixels[o + 1] = 0; pixels[o + 2] = 0; pixels[o + 3] = 0
        }
    }
    return ctx.makeImage()
}

func generateOutline(from maskImage: CGImage) -> CGImage? {
    let width = maskImage.width
    let height = maskImage.height
    let rect = CGRect(x: 0, y: 0, width: width, height: height)
    let glowRadius = CGFloat(min(30, max(4, min(width, height) / 100)))

    guard let silCtx = CGContext(
        data: nil, width: width, height: height,
        bitsPerComponent: 8, bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }

    silCtx.draw(maskImage, in: rect)
    silCtx.setBlendMode(.sourceIn)
    silCtx.setFillColor(UIColor.white.cgColor)
    silCtx.fill(rect)
    guard let whiteSilhouette = silCtx.makeImage() else { return nil }

    guard let outCtx = CGContext(
        data: nil, width: width, height: height,
        bitsPerComponent: 8, bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }

    outCtx.setShadow(offset: .zero, blur: glowRadius, color: UIColor.white.cgColor)
    outCtx.draw(whiteSilhouette, in: rect)
    outCtx.setShadow(offset: .zero, blur: 0, color: nil)
    outCtx.setBlendMode(.destinationOut)
    outCtx.draw(whiteSilhouette, in: rect)

    return outCtx.makeImage()
}

func composeMasks(_ masks: [CGImage]) -> CGImage? {
    guard let first = masks.first else { return nil }
    if masks.count == 1 { return first }

    let width = first.width
    let height = first.height
    guard let ctx = CGContext(
        data: nil, width: width, height: height,
        bitsPerComponent: 8, bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }

    let rect = CGRect(x: 0, y: 0, width: width, height: height)
    for mask in masks { ctx.draw(mask, in: rect) }
    return ctx.makeImage()
}

func processVisibleMasks(_ masks: [CGImage]) -> (binary: CGImage, outline: CGImage)? {
    guard let composed = composeMasks(masks),
          let binary = binarizeMask(composed),
          let outline = generateOutline(from: binary) else { return nil }
    return (binary, outline)
}

// MARK: - Glowing Outline View

struct GlowingOutlineView: View {
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
            .allowsHitTesting(false)
        }
    }
}

// MARK: - Lift Context Menu

struct LiftContextMenuView: View {
    let onCopy: () -> Void
    let onSave: () -> Void
    let onShare: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            Button { onCopy() } label: {
                Label("Copy", systemImage: "doc.on.doc")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16).padding(.vertical, 12)
                    .contentShape(Rectangle())
            }
            Divider()
            Button { onSave() } label: {
                Label("Save to Photos", systemImage: "square.and.arrow.down")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16).padding(.vertical, 12)
                    .contentShape(Rectangle())
            }
            Divider()
            Button { onShare() } label: {
                Label("Share...", systemImage: "square.and.arrow.up")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16).padding(.vertical, 12)
                    .contentShape(Rectangle())
            }
        }
        .foregroundColor(.primary).font(.body)
        .frame(width: 220)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .shadow(color: .black.opacity(0.2), radius: 20, y: 5)
    }
}

// MARK: - Lift Actions

func copyObject(_ image: UIImage) -> String {
    guard let pngData = image.pngData() else { return "Copy failed" }
    UIPasteboard.general.setData(pngData, forPasteboardType: UTType.png.identifier)
    return "Copied to clipboard"
}

func saveObject(_ image: UIImage, completion: @escaping (String) -> Void) {
    guard let pngData = image.pngData() else {
        completion("Save failed")
        return
    }
    PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
        guard status == .authorized || status == .limited else {
            DispatchQueue.main.async { completion("Photo access denied") }
            return
        }
        PHPhotoLibrary.shared().performChanges {
            let request = PHAssetCreationRequest.forAsset()
            request.addResource(with: .photo, data: pngData, options: nil)
        } completionHandler: { success, _ in
            DispatchQueue.main.async { completion(success ? "Saved to Photos" : "Save failed") }
        }
    }
}

// MARK: - Share Sheet

struct ActivityViewController: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

// MARK: - Toast View

struct ToastOverlay: View {
    let message: String?

    var body: some View {
        if let message = message {
            VStack {
                Spacer()
                Text(message)
                    .font(.subheadline).fontWeight(.medium)
                    .foregroundColor(.white)
                    .padding(.horizontal, 16).padding(.vertical, 10)
                    .background(Capsule().fill(Color.black.opacity(0.75)))
                    .padding(.bottom, 60)
            }
            .transition(.move(edge: .bottom).combined(with: .opacity))
        }
    }
}
