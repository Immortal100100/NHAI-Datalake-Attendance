package com.datalakeattendance.faceprocessor

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Bitmap.Config
import android.graphics.PointF
import android.graphics.Rect
import android.media.FaceDetector
import android.net.Uri
import android.util.Log
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.WritableArray
import com.facebook.react.bridge.WritableMap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.File
import kotlin.math.sqrt

/**
 * FaceAnalyzerModule — Android TurboModule (React Native New Architecture).
 *
 * Responsibilities (Phase 3 will fill in the TFLite bodies):
 *  1. loadModels()       — Load mediapipe_face_mesh.tflite + mobilefacenet_int8.tflite
 *                           from the Android assets folder into memory.
 *  2. analyzeFrame()     — Accept a YUV/RGBA pixel buffer, run both TFLite models,
 *                           return JSON with landmark coordinates + 128-D face vector.
 *  3. releaseModels()    — Dispose TFLite interpreters when app backgrounds.
 *
 * Phase 2: module skeleton + interface contract only.
 */
class FaceAnalyzerModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    companion object {
        const val NAME = "FaceAnalyzerModule"
        private const val TAG = "FaceAnalyzerModule"
        private const val FACE_MESH_MODEL   = "mediapipe_face_mesh.tflite"
        private const val FACE_NET_MODEL    = "mobilefacenet_int8.tflite"
        // 128-D embedding from fine-tuned w600k_mbf backbone (IMFDB, 100 Indian actors)
        private const val FACE_VECTOR_SIZE  = 128
        private const val SIMILARITY_THRESHOLD = 0.45f
    }

    // TFLite interpreters — initialized in loadModels() (Phase 3)
    // private var faceMeshInterpreter: Interpreter? = null
    private var faceNetInterpreter: Interpreter? = null

    override fun getName(): String = NAME

    // ─── Public API ───────────────────────────────────────────────────────────

    /**
     * Load both TFLite models from assets into native memory.
     * Must be called once before analyzeFrame().
     */
    @ReactMethod
    fun loadModels(promise: Promise) {
        try {
            ensureFaceNetInterpreter()
            promise.resolve("Models loaded")
        } catch (e: Exception) {
            Log.e(TAG, "loadModels failed", e)
            promise.reject("LOAD_MODEL_ERROR", e.message, e)
        }
    }

    /**
     * Analyze a single camera frame.
     *
     * @param width        Frame width in pixels
     * @param height       Frame height in pixels
     * @param pixelData    Raw RGBA or YUV_420_888 byte data as a Base64 string
     *                     (Phase 3 will accept this as a direct NativeBuffer handle)
     * @param runFaceNet   Whether to also run MobileFaceNet for embedding extraction
     * @param promise      Resolves with a WritableMap containing the analysis result
     *
     * Result shape:
     * {
     *   faceDetected: Boolean,
     *   landmarks: {
     *     leftEye:  Array<{x, y}>,   // 6 points for EAR
     *     rightEye: Array<{x, y}>,   // 6 points for EAR
     *     mouth:    Array<{x, y}>,   // 4 points for MAR
     *   } | null,
     *   faceVector: Array<Double> | null,   // 128 values
     *   processingMs: Int
     * }
     */
    @ReactMethod
    fun analyzeFrame(
        width: Int,
        height: Int,
        pixelData: String,
        runFaceNet: Boolean,
        promise: Promise
    ) {
        val startMs = System.currentTimeMillis()
        try {
            val bitmap = decodeBitmapFromPayload(pixelData)
            val faceBox = if (bitmap != null) detectFaceBox(bitmap) else null
            val hasFaceLikeInput = faceBox != null
            Log.d(TAG, "analyzeFrame runFaceNet=$runFaceNet bitmap=${bitmap != null} faceBox=$faceBox w=$width h=$height")

            val faceVector = if (runFaceNet && bitmap != null && faceBox != null) {
                val interpreter = faceNetInterpreter ?: ensureFaceNetInterpreter()
                if (interpreter != null) {
                    val faceCrop = cropFace(bitmap, faceBox)
                    try {
                        runFaceNetInference(interpreter, faceCrop)
                    } finally {
                        if (faceCrop !== bitmap) faceCrop.recycle()
                    }
                } else {
                    null
                }
            } else {
                null
            }

            val result: WritableMap = Arguments.createMap()
            result.putBoolean("faceDetected", hasFaceLikeInput)
            result.putNull("landmarks")
            // Face geometry (normalized 0..1) for offline liveness / head-movement check.
            if (faceBox != null && bitmap != null && bitmap.width > 0 && bitmap.height > 0) {
                val cx = (faceBox.left + faceBox.width() / 2f) / bitmap.width
                val cy = (faceBox.top + faceBox.height() / 2f) / bitmap.height
                val fw = faceBox.width().toFloat() / bitmap.width
                result.putDouble("faceCenterX", cx.toDouble())
                result.putDouble("faceCenterY", cy.toDouble())
                result.putDouble("faceWidth", fw.toDouble())
            }
            if (faceVector != null) {
                result.putArray("faceVector", toWritableArray(faceVector))
                Log.d(TAG, "analyzeFrame success faceVector=${faceVector.size}")
            } else {
                result.putNull("faceVector")
                Log.w(TAG, "analyzeFrame produced null faceVector")
            }
            result.putInt("processingMs", (System.currentTimeMillis() - startMs).toInt())
            promise.resolve(result)
        } catch (e: Exception) {
            Log.e(TAG, "analyzeFrame failed", e)
            promise.reject("ANALYZE_FRAME_ERROR", e.message, e)
        }
    }

    /**
     * Release TFLite interpreters and free native memory.
     * Call when the app enters background.
     */
    @ReactMethod
    fun releaseModels(promise: Promise) {
        // faceMeshInterpreter?.close(); faceMeshInterpreter = null
        faceNetInterpreter?.close()
        faceNetInterpreter = null
        promise.resolve(true)
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    /**
     * Load a TFLite model file from the Android assets directory into a ByteBuffer.
     * The returned buffer is direct (off-heap) and ready for TFLite interpreter init.
     */
    private fun loadModelFromAssets(context: Context, modelName: String): ByteBuffer {
        val assetManager = context.assets
        val inputStream  = assetManager.open(modelName)
        val bytes        = inputStream.readBytes()
        inputStream.close()
        val buffer = ByteBuffer.allocateDirect(bytes.size)
        buffer.put(bytes)
        buffer.rewind()
        return buffer
    }

    private fun ensureFaceNetInterpreter(): Interpreter? {
        if (faceNetInterpreter != null) return faceNetInterpreter
        return try {
            val faceNetBuffer = loadModelFromAssets(reactContext, FACE_NET_MODEL)
            val opts = Interpreter.Options().apply {
                setNumThreads(2)
                setUseXNNPACK(true)
            }
            faceNetInterpreter = Interpreter(faceNetBuffer, opts)
            Log.i(TAG, "MobileFaceNet interpreter initialized")
            faceNetInterpreter
        } catch (e: Exception) {
            Log.e(TAG, "ensureFaceNetInterpreter failed", e)
            null
        }
    }

    /**
     * Expected Phase 4 interim payload from JS:
     *   {"uri":"file:///.../photo.jpg","source":"camera_photo"}
     */
    private fun decodeBitmapFromPayload(pixelData: String): Bitmap? {
        if (pixelData.isBlank()) return null
        return try {
            val json = JSONObject(pixelData)
            val uriText = json.optString("uri", "")
            if (uriText.isBlank()) return null
            val uri = Uri.parse(uriText)

            when (uri.scheme?.lowercase()) {
                "file", null -> {
                    val path = uri.path ?: return null
                    BitmapFactory.decodeFile(File(path).absolutePath)
                }
                "content" -> {
                    val input = reactContext.contentResolver.openInputStream(uri) ?: return null
                    input.use { BitmapFactory.decodeStream(it) }
                }
                else -> null
            }
        } catch (_: Exception) {
            null
        }
    }

    /**
     * Detect the largest frontal face and return an expanded square crop box
     * (in original-bitmap pixel coordinates). Returns null if no face is found.
     *
     * Cropping to the face before inference is critical: the recognition model
     * was trained on tightly-cropped aligned faces. Feeding a full photo makes
     * the embedding depend on background + framing, which is extremely sensitive
     * to lighting/environment and causes genuine matches to be rejected.
     */
    private fun detectFaceBox(bitmap: Bitmap): Rect? {
        return try {
            // android.media.FaceDetector requires an RGB_565 bitmap with even width.
            val src = if (bitmap.config == Config.RGB_565 && bitmap.isMutable) {
                bitmap
            } else {
                bitmap.copy(Config.RGB_565, true)
            }
            val faces = arrayOfNulls<FaceDetector.Face>(1)
            val detector = FaceDetector(src.width, src.height, 1)
            val found = detector.findFaces(src, faces)
            val face = faces[0]
            val box = if (found > 0 && face != null) {
                val mid = PointF()
                face.getMidPoint(mid)
                val eye = face.eyesDistance()
                // Face width ~= 2x eye distance; use ~3x for a full head+margin square crop.
                val side = eye * 3.0f
                val cx = mid.x
                // Shift center down from the eye-line toward nose/mouth for a centered face.
                val cy = mid.y + eye * 0.5f
                val left = (cx - side / 2f).toInt().coerceAtLeast(0)
                val top = (cy - side / 2f).toInt().coerceAtLeast(0)
                val right = (cx + side / 2f).toInt().coerceAtMost(src.width)
                val bottom = (cy + side / 2f).toInt().coerceAtMost(src.height)
                if (right - left >= 24 && bottom - top >= 24) {
                    Rect(left, top, right, bottom)
                } else {
                    null
                }
            } else {
                null
            }
            if (src !== bitmap) src.recycle()
            box
        } catch (e: Exception) {
            Log.w(TAG, "detectFaceBox failed", e)
            null
        }
    }

    private fun cropFace(bitmap: Bitmap, box: Rect): Bitmap {
        val l = box.left.coerceIn(0, bitmap.width - 1)
        val t = box.top.coerceIn(0, bitmap.height - 1)
        val w = box.width().coerceAtMost(bitmap.width - l)
        val h = box.height().coerceAtMost(bitmap.height - t)
        if (w <= 0 || h <= 0) return bitmap
        return Bitmap.createBitmap(bitmap, l, t, w, h)
    }

    private fun runFaceNetInference(interpreter: Interpreter, bitmap: Bitmap): FloatArray {
        val inTensor = interpreter.getInputTensor(0)
        val inShape = inTensor.shape() // e.g. [1,128,128,3] or [1,112,112,3]
        val inType = inTensor.dataType()
        val h = inShape.getOrNull(1) ?: 128
        val w = inShape.getOrNull(2) ?: 128
        val c = inShape.getOrNull(3) ?: 3
        require(c == 3) { "Expected RGB input channels=3, got $c" }

        val resized = Bitmap.createScaledBitmap(bitmap, w, h, true)
        val pixels = IntArray(w * h)
        resized.getPixels(pixels, 0, w, 0, 0, w, h)

        val inputBuffer = when (inType) {
            DataType.FLOAT32 -> {
                val bb = ByteBuffer.allocateDirect(4 * w * h * c).order(ByteOrder.nativeOrder())
                for (px in pixels) {
                    val r = (px shr 16) and 0xff
                    val g = (px shr 8) and 0xff
                    val b = px and 0xff
                    // InsightFace normalization: [0,255] -> [0,1] -> [-1,1]
                    bb.putFloat((r / 255f - 0.5f) / 0.5f)
                    bb.putFloat((g / 255f - 0.5f) / 0.5f)
                    bb.putFloat((b / 255f - 0.5f) / 0.5f)
                }
                bb.rewind()
                bb
            }
            DataType.UINT8, DataType.INT8 -> {
                val q = inTensor.quantizationParams()
                val scale = if (q.scale == 0f) 1f else q.scale
                val zp = q.zeroPoint
                val bb = ByteBuffer.allocateDirect(w * h * c).order(ByteOrder.nativeOrder())
                for (px in pixels) {
                    val rgb = intArrayOf(
                        (px shr 16) and 0xff,
                        (px shr 8) and 0xff,
                        px and 0xff
                    )
                    for (v in rgb) {
                        val f = (v / 255f - 0.5f) / 0.5f
                        val qv = (f / scale + zp).toInt().coerceIn(-128, 255)
                        bb.put(qv.toByte())
                    }
                }
                bb.rewind()
                bb
            }
            else -> throw IllegalStateException("Unsupported input type: $inType")
        }

        val outTensor = interpreter.getOutputTensor(0)
        val outType = outTensor.dataType()
        val outShape = outTensor.shape()
        val outSize = outShape.fold(1) { acc, v -> acc * v }
        val outBytes = outTensor.numBytes()

        val embedding = when (outType) {
            DataType.FLOAT32 -> {
                // Use a raw output buffer so tensor shapes like [1,128] and [128]
                // are both accepted by TFLite without Java array shape mismatch.
                val outBuffer = ByteBuffer.allocateDirect(outBytes).order(ByteOrder.nativeOrder())
                outBuffer.rewind()
                interpreter.run(inputBuffer, outBuffer)
                outBuffer.rewind()
                val out = FloatArray(outSize)
                outBuffer.asFloatBuffer().get(out, 0, outSize)
                out
            }
            DataType.UINT8, DataType.INT8 -> {
                val outBuffer = ByteBuffer.allocateDirect(outBytes).order(ByteOrder.nativeOrder())
                outBuffer.rewind()
                interpreter.run(inputBuffer, outBuffer)
                outBuffer.rewind()
                val outRaw = ByteArray(outSize)
                outBuffer.get(outRaw, 0, outSize)
                val q = outTensor.quantizationParams()
                val scale = if (q.scale == 0f) 1f else q.scale
                val zp = q.zeroPoint
                FloatArray(outSize) { i ->
                    val raw = if (outType == DataType.UINT8) {
                        outRaw[i].toInt() and 0xff
                    } else {
                        outRaw[i].toInt()
                    }
                    ((raw - zp) * scale)
                }
            }
            else -> throw IllegalStateException("Unsupported output type: $outType")
        }

        // Squeeze leading batch if present
        val vec = if (embedding.size == FACE_VECTOR_SIZE) {
            embedding
        } else if (embedding.size > FACE_VECTOR_SIZE) {
            embedding.copyOfRange(0, FACE_VECTOR_SIZE)
        } else {
            val out = FloatArray(FACE_VECTOR_SIZE)
            for (i in embedding.indices) out[i] = embedding[i]
            out
        }

        return l2Normalize(vec)
    }

    private fun l2Normalize(vec: FloatArray): FloatArray {
        var normSq = 0.0
        for (v in vec) normSq += (v * v).toDouble()
        val norm = sqrt(normSq).toFloat().coerceAtLeast(1e-8f)
        val out = FloatArray(vec.size)
        for (i in vec.indices) out[i] = vec[i] / norm
        return out
    }

    private fun toWritableArray(values: FloatArray): WritableArray {
        val arr = Arguments.createArray()
        for (v in values) arr.pushDouble(v.toDouble())
        return arr
    }
}
