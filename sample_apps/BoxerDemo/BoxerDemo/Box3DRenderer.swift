import SwiftUI
import RealityKit
import simd

/// Projects 3D bounding boxes onto screen using the voxel-frame coordinate system.
/// Boxer voxel frame: X = right, Y = up, Z = forward (depth).
struct Box3DOverlay: View {
    let detections: [Detection3D]
    let userBoxes: [CGRect]
    let viewSize: CGSize
    // Camera intrinsics for projection (scaled to view size)
    var fx: CGFloat = 720
    var fy: CGFloat = 720

    static let colors: [Color] = [.cyan, .green, .orange, .pink, .yellow, .blue, .red, .purple]

    var body: some View {
        Canvas { ctx, size in
            let cx = size.width / 2
            let cy = size.height / 2

            for (di, det) in detections.enumerated() {
                let color = Self.colors[di % Self.colors.count]
                let corners = boxCorners(det)

                // Project 3D corners to 2D screen
                let projected: [CGPoint?] = corners.map { pt in
                    // Boxer voxel: X=right, Y=up, Z=forward
                    let z = CGFloat(pt.z)
                    guard z > 0.05 else { return nil }
                    let sx = CGFloat(pt.x) / z * fx + cx
                    let sy = -CGFloat(pt.y) / z * fy + cy  // negate Y: up in 3D → down on screen
                    return CGPoint(x: sx, y: sy)
                }

                // Draw edges
                let edges = [
                    (0,1),(1,2),(2,3),(3,0),  // front face
                    (4,5),(5,6),(6,7),(7,4),  // back face
                    (0,4),(1,5),(2,6),(3,7),  // connecting
                ]
                for (a, b) in edges {
                    guard let pa = projected[a], let pb = projected[b],
                          inBounds(pa, size) || inBounds(pb, size) else { continue }
                    var path = Path()
                    path.move(to: pa); path.addLine(to: pb)
                    // Front face thicker
                    let lw: CGFloat = (a < 4 && b < 4) ? 3.0 : 1.5
                    ctx.stroke(path, with: .color(color), lineWidth: lw)
                }

                // Fill front face
                let frontPts = [0,1,2,3].compactMap { projected[$0] }
                if frontPts.count == 4 {
                    var fp = Path()
                    fp.move(to: frontPts[0])
                    for p in frontPts.dropFirst() { fp.addLine(to: p) }
                    fp.closeSubpath()
                    ctx.fill(fp, with: .color(color.opacity(0.12)))
                }

                // Fill top face
                let topPts = [3,2,6,7].compactMap { projected[$0] }
                if topPts.count == 4 {
                    var tp = Path()
                    tp.move(to: topPts[0])
                    for p in topPts.dropFirst() { tp.addLine(to: p) }
                    tp.closeSubpath()
                    ctx.fill(tp, with: .color(color.opacity(0.06)))
                }

                // Label at front face center
                if let p0 = projected[0], let p2 = projected[2] {
                    let labelPt = CGPoint(x: (p0.x + p2.x) / 2, y: p0.y - 16)
                    let label = String(format: "%.2f×%.2f×%.2f m  %.1fm",
                                       det.size.x, det.size.y, det.size.z, det.distance)
                    // Background
                    let textW: CGFloat = CGFloat(label.count) * 6.5
                    let pillRect = CGRect(x: labelPt.x - textW/2 - 4, y: labelPt.y - 10, width: textW + 8, height: 20)
                    ctx.fill(Path(roundedRect: pillRect, cornerRadius: 6), with: .color(color.opacity(0.85)))
                    ctx.draw(
                        Text(label).font(.system(size: 10, weight: .bold, design: .monospaced)).foregroundColor(.white),
                        at: labelPt
                    )
                }
            }
        }
        .allowsHitTesting(false)
    }

    /// 8 corners of a 3D OBB in voxel frame.
    private func boxCorners(_ det: Detection3D) -> [SIMD3<Float>] {
        let c = det.center
        let hh = det.size.x / 2  // half height (Y in voxel)
        let hw = det.size.y / 2  // half width (X in voxel)
        let hd = det.size.z / 2  // half depth (Z in voxel)
        let cosY = cos(det.yaw), sinY = sin(det.yaw)

        // Local box corners: X=width, Y=height, Z=depth
        let local: [SIMD3<Float>] = [
            // Front face (closer, Z = center.z - hd)
            SIMD3(-hw, -hh, -hd), SIMD3( hw, -hh, -hd),
            SIMD3( hw,  hh, -hd), SIMD3(-hw,  hh, -hd),
            // Back face (farther, Z = center.z + hd)
            SIMD3(-hw, -hh,  hd), SIMD3( hw, -hh,  hd),
            SIMD3( hw,  hh,  hd), SIMD3(-hw,  hh,  hd),
        ]

        return local.map { lc in
            // Rotate by yaw around Y axis (vertical in voxel frame)
            let rx = lc.x * cosY + lc.z * sinY
            let rz = -lc.x * sinY + lc.z * cosY
            return SIMD3(c.x + rx, c.y + lc.y, c.z + rz)
        }
    }

    private func inBounds(_ p: CGPoint, _ size: CGSize) -> Bool {
        p.x > -300 && p.x < size.width + 300 && p.y > -300 && p.y < size.height + 300
    }
}
