import UIKit

/// Visualization helpers: turbo colormap for depth, RGB encoding for normals.
/// All methods accept the full 504x504 model output and a valid rect that
/// describes where the actual image content sits (letterbox region). The
/// returned UIImage is cropped to the valid rect so it matches the original
/// aspect ratio.
enum Visualization {

    /// Render a metric depth map as a turbo-colormap UIImage.
    static func depthImage(
        _ depth: [Float], size: Int, dMin: Float, dMax: Float,
        validX: Int, validY: Int, validW: Int, validH: Int
    ) -> UIImage {
        var rgba = [UInt8](repeating: 0, count: validW * validH * 4)
        let span = max(dMax - dMin, 1e-3)
        for row in 0..<validH {
            let srcRow = (validY + row) * size + validX
            let dstRow = row * validW
            for col in 0..<validW {
                let d = depth[srcRow + col]
                if d <= 0 { continue }
                let t = min(max((d - dMin) / span, 0), 1)
                let (r, g, b) = turbo(1 - t)
                let idx = (dstRow + col) * 4
                rgba[idx] = UInt8(r * 255)
                rgba[idx + 1] = UInt8(g * 255)
                rgba[idx + 2] = UInt8(b * 255)
                rgba[idx + 3] = 255
            }
        }
        return makeUIImage(rgba: rgba, width: validW, height: validH)
    }

    /// Render surface normals as an RGB image.
    static func normalImage(
        _ normal: [Float], mask: [Float], size: Int,
        validX: Int, validY: Int, validW: Int, validH: Int
    ) -> UIImage {
        var rgba = [UInt8](repeating: 0, count: validW * validH * 4)
        for row in 0..<validH {
            let srcRow = (validY + row) * size + validX
            let dstRow = row * validW
            for col in 0..<validW {
                let si = srcRow + col
                if mask[si] < 0.5 { continue }
                let nx = normal[si * 3]
                let ny = -normal[si * 3 + 1]
                let nz = normal[si * 3 + 2]
                let idx = (dstRow + col) * 4
                rgba[idx] = UInt8(((nx + 1) * 0.5 * 255).rounded())
                rgba[idx + 1] = UInt8(((ny + 1) * 0.5 * 255).rounded())
                rgba[idx + 2] = UInt8(((nz + 1) * 0.5 * 255).rounded())
                rgba[idx + 3] = 255
            }
        }
        return makeUIImage(rgba: rgba, width: validW, height: validH)
    }

    // MARK: - Turbo colormap (Mikhailov 2019)
    private static func turbo(_ t: Float) -> (Float, Float, Float) {
        let x = min(max(t, 0), 1)
        let r =  0.13572138 + x * ( 4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x *  59.28637943))))
        let g =  0.09140261 + x * ( 2.19418839 + x * (  4.84296658 + x * (-14.18503333 + x * (  4.27729857 + x *   2.82956604))))
        let b =  0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * ( -89.90310912 + x *  27.34824973))))
        return (min(max(r, 0), 1), min(max(g, 0), 1), min(max(b, 0), 1))
    }

    private static func makeUIImage(rgba: [UInt8], width: Int, height: Int) -> UIImage {
        var data = rgba
        let ctx = CGContext(
            data: &data,
            width: width, height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        return UIImage(cgImage: ctx.makeImage()!)
    }
}
