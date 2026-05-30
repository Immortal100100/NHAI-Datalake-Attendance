import Foundation

/**
 * FaceAnalyzerModule — iOS Swift TurboModule for offline TFLite face analysis.
 *
 * Responsibilities (Phase 3 fills the TFLite bodies):
 *  1. loadModels()     — Load mediapipe_face_mesh.tflite + mobilefacenet_int8.tflite
 *                         from the app bundle into TFLite Interpreter instances.
 *  2. analyzeFrame()   — Run MediaPipe face mesh detection + optional MobileFaceNet
 *                         embedding on an incoming CVPixelBuffer / Data payload.
 *  3. releaseModels()  — Free TFLite Interpreter memory on app background.
 *
 * Phase 2: class skeleton + interface contract only.
 */
@objc(FaceAnalyzerModule)
class FaceAnalyzerModule: NSObject {

    // TFLite interpreters — initialized in loadModels() (Phase 3)
    // private var faceMeshInterpreter: Interpreter?
    // private var faceNetInterpreter:  Interpreter?

    private let MODEL_FACE_MESH = "mediapipe_face_mesh"
    private let MODEL_FACE_NET  = "mobilefacenet_int8"
    private let MODEL_EXT       = "tflite"

    // MARK: - Module Registration

    @objc static func requiresMainQueueSetup() -> Bool { false }

    // MARK: - Public API

    /**
     * Load both TFLite models from the iOS app bundle.
     * Call once, e.g. in AppDelegate or when the Camera screen mounts.
     *
     * Phase 3 implementation:
     *   let options = Interpreter.Options(); options.threadCount = 2
     *   faceMeshInterpreter = try Interpreter(modelPath: meshPath, options: options)
     *   faceNetInterpreter  = try Interpreter(modelPath: netPath,  options: options)
     *   try faceMeshInterpreter?.allocateTensors()
     *   try faceNetInterpreter?.allocateTensors()
     */
    @objc func loadModels(
        _ resolve: @escaping RCTPromiseResolveBlock,
        reject: @escaping RCTPromiseRejectBlock
    ) {
        guard
            Bundle.main.path(forResource: MODEL_FACE_MESH, ofType: MODEL_EXT) != nil,
            Bundle.main.path(forResource: MODEL_FACE_NET,  ofType: MODEL_EXT) != nil
        else {
            // Models not yet bundled — expected during Phase 2
            resolve("Models not found in bundle (stub mode)")
            return
        }
        // Phase 3: initialize TFLite interpreters here
        resolve("Models loaded (stub)")
    }

    /**
     * Analyze a single camera frame.
     *
     * @param width       Frame width
     * @param height      Frame height
     * @param pixelData   BGRA or YUV byte string (Phase 3 uses CVPixelBuffer via NativeBuffer)
     * @param runFaceNet  Whether to also run the MobileFaceNet embedding pass
     *
     * Resolves with NSDictionary matching the shape defined in FaceProcessor.ts:
     * {
     *   faceDetected: Bool,
     *   landmarks: { leftEye, rightEye, mouth } | NSNull,
     *   faceVector: [Double] | NSNull,
     *   processingMs: Int
     * }
     */
    @objc func analyzeFrame(
        _ width: NSNumber,
        height: NSNumber,
        pixelData: String,
        runFaceNet: Bool,
        resolve: @escaping RCTPromiseResolveBlock,
        reject: @escaping RCTPromiseRejectBlock
    ) {
        let startTime = CFAbsoluteTimeGetCurrent()
        // Phase 3 implementation:
        // 1. Decode pixelData → CVPixelBuffer
        // 2. Create MLImage / TFLite input tensor
        // 3. faceMeshInterpreter.copy(inputData, toInputAt: 0)
        //    faceMeshInterpreter.invoke()
        //    let output = faceMeshInterpreter.output(at: 0) → [1,1,468,3] float32
        // 4. Extract EAR keypoints (left: 33,160,158,133,153,144; right: 362,385,387,263,373,380)
        //    Extract MAR keypoints (61,291,0,17)
        // 5. If runFaceNet: crop face bbox → faceNetInterpreter → 512-D vector (w600k_mbf)
        let processingMs = Int((CFAbsoluteTimeGetCurrent() - startTime) * 1000)

        resolve([
            "faceDetected": false,
            "landmarks":    NSNull(),
            "faceVector":   NSNull(),
            "processingMs": processingMs,
        ] as [String: Any])
    }

    /**
     * Release TFLite interpreters and free native memory.
     */
    @objc func releaseModels(
        _ resolve: @escaping RCTPromiseResolveBlock,
        reject: @escaping RCTPromiseRejectBlock
    ) {
        // faceMeshInterpreter = nil
        // faceNetInterpreter  = nil
        resolve(true)
    }
}
