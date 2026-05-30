/**
 * FaceProcessor.ts — JS bridge interface for the native TFLite face analysis pipeline.
 *
 * Architecture (VisionCamera v5 + Nitro Modules):
 *   Camera frame output (CameraFrameOutput)
 *     → worklet (react-native-vision-camera-worklets, Phase 3)
 *       → MediaPipe Face Mesh TFLite  → 468 landmarks
 *       → MobileFaceNet INT8 TFLite   → 128-D face vector
 *     → JS thread                     → UI state update
 *
 * Phase 2: types + stub. Phase 3 wires real inference.
 */
import { NativeModules, Platform } from 'react-native';

// ─── Result Types ─────────────────────────────────────────────────────────────

/** 2-D pixel coordinate in the camera frame. */
export interface Landmark {
  x: number;
  y: number;
}

/**
 * Indices into the MediaPipe Face Mesh 468-point model that
 * are relevant for EAR (blink) and MAR (smile) calculations.
 */
export interface FaceLandmarks {
  // Left eye:  p1..p6 for EAR
  leftEye: [Landmark, Landmark, Landmark, Landmark, Landmark, Landmark];
  // Right eye: p1..p6 for EAR
  rightEye: [Landmark, Landmark, Landmark, Landmark, Landmark, Landmark];
  // Mouth outer corners + top/bottom for MAR
  mouth: [Landmark, Landmark, Landmark, Landmark];
}

/** Full result returned by the native frame processor on every camera frame. */
export interface FaceProcessorResult {
  /** True when a face bounding box is detected in the frame. */
  faceDetected: boolean;
  /** Extracted landmark coordinates (only valid when faceDetected is true). */
  landmarks: FaceLandmarks | null;
  /**
   * 128-dimensional MobileFaceNet embedding vector.
   * Populated only during enrollment or active verification pass.
   */
  faceVector: number[] | null;
  /** Normalized (0..1) face-box center X within the frame (for head-movement liveness). */
  faceCenterX?: number;
  /** Normalized (0..1) face-box center Y within the frame. */
  faceCenterY?: number;
  /** Normalized (0..1) face-box width within the frame. */
  faceWidth?: number;
  /** Frame processing latency in milliseconds (for perf benchmarking). */
  processingMs: number;
}

export interface FaceAnalyzerNativeModule {
  loadModels(): Promise<string>;
  analyzeFrame(
    width: number,
    height: number,
    pixelData: string,
    runFaceNet: boolean,
  ): Promise<FaceProcessorResult>;
  releaseModels(): Promise<boolean>;
}

const nativeModule = (NativeModules.FaceAnalyzerModule ??
  null) as FaceAnalyzerNativeModule | null;

export const NATIVE_ANALYZER_AVAILABLE = nativeModule !== null;

export async function loadNativeFaceModels(): Promise<void> {
  if (!nativeModule) return;
  await nativeModule.loadModels();
}

export async function analyzeFrameNative(params: {
  width: number;
  height: number;
  pixelData: string;
  runFaceNet?: boolean;
}): Promise<FaceProcessorResult | null> {
  if (!nativeModule) return null;
  const result = await nativeModule.analyzeFrame(
    params.width,
    params.height,
    params.pixelData,
    params.runFaceNet ?? true,
  );
  return result;
}

export async function releaseNativeFaceModels(): Promise<void> {
  if (!nativeModule) return;
  await nativeModule.releaseModels();
}

// ─── Stub (Phase 2) ──────────────────────────────────────────────────────────

/**
 * Returns an empty/stub result.
 * Replaced in Phase 3 by a real worklet that calls the TFLite native module.
 */
export function processFaceFrameStub(): FaceProcessorResult {
  return {
    faceDetected: false,
    landmarks: null,
    faceVector: null,
    processingMs: 0,
  };
}

// ─── Constants ───────────────────────────────────────────────────────────────

/** MediaPipe Face Mesh model filename (to be bundled in native assets). */
export const FACE_MESH_MODEL = 'mediapipe_face_mesh.tflite';

/** MobileFaceNet INT8 quantized model filename. */
export const FACE_NET_MODEL = 'mobilefacenet_int8.tflite';

/** Minimum cosine similarity to accept as a match.
 * Calibrated on-device: genuine ~0.55, impostor ~0.14 (face-crop pipeline). */
export const SIMILARITY_THRESHOLD = 0.45;
/** Minimum top1-top2 margin (open-set gating; matters when gallery has 2+ users). */
export const TOP1_TOP2_MARGIN_MIN = 0.08;
/** Number of frames to average for robust verification. */
export const VERIFICATION_WINDOW_FRAMES = 5;

/**
 * Offline liveness (anti-spoofing) tuning.
 * The hackathon requires a basic active challenge (blink/smile/turn head).
 * We use a head-movement challenge: across the capture window the face must
 * move enough (turn/shift) to indicate a live subject rather than a static photo.
 */
/** Minimum spread of normalized face-center X across frames to count as head movement. */
export const LIVENESS_MIN_CENTER_X_RANGE = 0.06;
/** Minimum combined movement (X range + Y range + width range) across frames. */
export const LIVENESS_MIN_TOTAL_MOVEMENT = 0.08;

export interface FaceMotionSample {
  centerX?: number;
  centerY?: number;
  width?: number;
}

export interface LivenessResult {
  passed: boolean;
  centerXRange: number;
  centerYRange: number;
  widthRange: number;
  totalMovement: number;
}

/**
 * Decide liveness from per-frame face geometry collected during the capture window.
 * A live person naturally moves (and is asked to turn their head slightly); a held
 * photo/screen tends to stay rigid, producing near-zero movement.
 */
export function evaluateLiveness(samples: FaceMotionSample[]): LivenessResult {
  const xs = samples.map(s => s.centerX).filter((v): v is number => typeof v === 'number');
  const ys = samples.map(s => s.centerY).filter((v): v is number => typeof v === 'number');
  const ws = samples.map(s => s.width).filter((v): v is number => typeof v === 'number');

  const range = (arr: number[]): number =>
    arr.length >= 2 ? Math.max(...arr) - Math.min(...arr) : 0;

  const centerXRange = range(xs);
  const centerYRange = range(ys);
  const widthRange = range(ws);
  const totalMovement = centerXRange + centerYRange + widthRange;

  const passed =
    xs.length >= 2 &&
    (centerXRange >= LIVENESS_MIN_CENTER_X_RANGE ||
      totalMovement >= LIVENESS_MIN_TOTAL_MOVEMENT);

  return { passed, centerXRange, centerYRange, widthRange, totalMovement };
}

/**
 * Number of float values in a face embedding.
 * 512 = InsightFace w600k_mbf (WebFace600K pre-trained, used in production).
 * 128 = Custom synthetic model (used only during development).
 */
export const FACE_VECTOR_SIZE = 128;

// ─── Matching Helpers (Phase 3 integration) ─────────────────────────────────

export interface MatchPolicyResult {
  matched: boolean;
  userId: string | null;
  score: number;
  margin: number;
}

export interface GalleryProfile {
  id: string;
  faceVector: number[];
}

export function l2Normalize(vec: number[]): number[] {
  if (vec.length === 0) return vec;
  const norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0));
  if (norm < 1e-8) return vec.map(() => 0);
  return vec.map(v => v / norm);
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const den = Math.sqrt(na) * Math.sqrt(nb);
  return den < 1e-8 ? 0 : dot / den;
}

export function averageEmbeddings(frames: number[][]): number[] {
  if (frames.length === 0) return [];
  const dim = frames[0].length;
  const out = new Array<number>(dim).fill(0);
  for (const emb of frames) {
    for (let i = 0; i < dim; i += 1) out[i] += emb[i] ?? 0;
  }
  for (let i = 0; i < dim; i += 1) out[i] /= frames.length;
  return l2Normalize(out);
}

export function matchAgainstGallery(
  probeEmbedding: number[],
  gallery: GalleryProfile[],
  threshold: number = SIMILARITY_THRESHOLD,
  marginMin: number = TOP1_TOP2_MARGIN_MIN,
): MatchPolicyResult {
  const valid = gallery.filter(
    p => Array.isArray(p.faceVector) && p.faceVector.length === FACE_VECTOR_SIZE,
  );
  if (valid.length === 0 || probeEmbedding.length !== FACE_VECTOR_SIZE) {
    return { matched: false, userId: null, score: 0, margin: 0 };
  }

  const probe = l2Normalize(probeEmbedding);
  const scored = valid
    .map(p => ({
      userId: p.id,
      score: cosineSimilarity(probe, l2Normalize(p.faceVector)),
    }))
    .sort((a, b) => b.score - a.score);

  const top1 = scored[0];
  const top2 = scored[1];
  const margin = top2 ? top1.score - top2.score : top1.score;
  const accepted = top1.score >= threshold && margin >= marginMin;
  return {
    matched: accepted,
    userId: accepted ? top1.userId : null,
    score: top1.score,
    margin,
  };
}

// Deterministic synthetic vector for Phase 3 UI wiring only.
// Replace in Phase 4 with native TFLite inference outputs.
export function syntheticFaceVector(seedText: string): number[] {
  let seed = 2166136261;
  for (let i = 0; i < seedText.length; i += 1) {
    seed ^= seedText.charCodeAt(i);
    seed = Math.imul(seed, 16777619) >>> 0;
  }
  const out = new Array<number>(FACE_VECTOR_SIZE);
  for (let i = 0; i < FACE_VECTOR_SIZE; i += 1) {
    seed = (1664525 * seed + 1013904223) >>> 0;
    out[i] = (seed / 0xffffffff) * 2 - 1;
  }
  return l2Normalize(out);
}

/**
 * Creates a slightly perturbed embedding around a base vector.
 * Useful for Phase 3 policy simulation before real frame embeddings are available.
 */
export function jitterEmbedding(
  baseVector: number[],
  seedText: string,
  noiseScale: number = 0.03,
): number[] {
  if (baseVector.length !== FACE_VECTOR_SIZE) return [];
  let seed = 2166136261;
  for (let i = 0; i < seedText.length; i += 1) {
    seed ^= seedText.charCodeAt(i);
    seed = Math.imul(seed, 16777619) >>> 0;
  }
  const out = new Array<number>(FACE_VECTOR_SIZE);
  for (let i = 0; i < FACE_VECTOR_SIZE; i += 1) {
    seed = (1664525 * seed + 1013904223) >>> 0;
    const rnd = (seed / 0xffffffff) * 2 - 1;
    out[i] = baseVector[i] + rnd * noiseScale;
  }
  return l2Normalize(out);
}
