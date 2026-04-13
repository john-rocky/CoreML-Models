import ARKit
import Combine

/// Manages ARKit session and provides frame data for Boxer inference.
final class ARSessionManager: NSObject, ObservableObject, ARSessionDelegate {
    let session = ARSession()

    @Published var currentFrame: ARFrame?
    @Published var isRunning = false
    @Published var hasDepth = false

    override init() {
        super.init()
        session.delegate = self
    }

    func start() {
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = []

        // Enable scene depth if LiDAR is available
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
            hasDepth = true
        }

        session.run(config)
        isRunning = true
    }

    func stop() {
        session.pause()
        isRunning = false
    }

    /// Capture current frame snapshot with all metadata for inference.
    func captureFrame() -> ARFrameData? {
        guard let frame = currentFrame else { return nil }
        let camera = frame.camera

        // Camera intrinsics: 3x3 matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
        let intrinsics = camera.intrinsics
        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let cx = intrinsics[2][0]
        let cy = intrinsics[2][1]

        // Camera pose: 4x4 transform (world-to-camera is inverse of camera.transform)
        // ARKit: camera.transform = T_world_camera (camera position in world)
        let T_world_camera = camera.transform  // simd_float4x4

        // Image resolution
        let imageResolution = camera.imageResolution  // CGSize

        // Scene depth (LiDAR)
        let depthMap = frame.sceneDepth?.depthMap

        // Captured image
        let pixelBuffer = frame.capturedImage

        return ARFrameData(
            pixelBuffer: pixelBuffer,
            depthMap: depthMap,
            intrinsics: CameraIntrinsics(fx: fx, fy: fy, cx: cx, cy: cy),
            T_world_camera: T_world_camera,
            imageWidth: Int(imageResolution.width),
            imageHeight: Int(imageResolution.height)
        )
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        DispatchQueue.main.async {
            self.currentFrame = frame
        }
    }
}

// MARK: - Data Types

struct CameraIntrinsics {
    let fx: Float
    let fy: Float
    let cx: Float
    let cy: Float
}

struct ARFrameData {
    let pixelBuffer: CVPixelBuffer
    let depthMap: CVPixelBuffer?
    let intrinsics: CameraIntrinsics
    let T_world_camera: simd_float4x4
    let imageWidth: Int
    let imageHeight: Int

    /// Extract gravity-aligned rotation matrix.
    /// ARKit world frame: Y-up, gravity = -Y.
    /// Boxer voxel frame: Z-up, gravity = -Z.
    /// We need T_voxel_camera for Plucker rays.
    var gravityAlignedRotation: simd_float3x3 {
        // ARKit's world frame is already gravity-aligned (Y-up).
        // Extract just the rotation from T_world_camera.
        let R = simd_float3x3(
            simd_float3(T_world_camera.columns.0.x, T_world_camera.columns.0.y, T_world_camera.columns.0.z),
            simd_float3(T_world_camera.columns.1.x, T_world_camera.columns.1.y, T_world_camera.columns.1.z),
            simd_float3(T_world_camera.columns.2.x, T_world_camera.columns.2.y, T_world_camera.columns.2.z)
        )
        return R
    }
}
