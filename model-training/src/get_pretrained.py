"""
get_pretrained.py — Download InsightFace's pre-trained MobileFaceNet
(w600k_mbf, trained on WebFace600K — 600K diverse identities including
Indian/South-Asian faces) and prepare it for INT8 TFLite export.

Run this in the nhai-export conda environment (Python 3.11):
  conda activate nhai-export
  pip install insightface
  $env:PYTHONUTF8="1"; $env:TF_ENABLE_ONEDNN_OPTS="0"
  $py = "C:\\Users\\kunal\\miniconda3\\envs\\nhai-export\\python.exe"
  & $py src/get_pretrained.py

The script will:
  1. Auto-download buffalo_sc (w600k_mbf.onnx, ~16 MB)
  2. Copy + simplify to exports/w600k_mbf.onnx
  3. INT8 quantize  -> exports/w600k_mbf_int8.onnx  (~4 MB)
  4. Convert        -> exports/w600k_mbf_int8.tflite (~4 MB) [PASS < 20 MB]
  5. Copy to Android/iOS assets

Model specs:
  Input  : [1, 3, 112, 112]  float32  (normalised mean=0.5, std=0.5)
  Output : [1, 512]           float32  L2-normalised face embedding
  Note   : embedding_dim=512 (vs 128 in synthetic model — update constants below)
"""

from __future__ import annotations

import io
import os
import shutil
import sys
from pathlib import Path

# Force UTF-8 for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np

# ─── Config ──────────────────────────────────────────────────────────────────

CONFIG = {
    "model_name":      "buffalo_sc",
    "onnx_filename":   "w600k_mbf.onnx",
    "onnx_out":        "exports/w600k_mbf.onnx",
    "onnx_int8_out":   "exports/w600k_mbf_int8.onnx",
    "tflite_out":      "exports/w600k_mbf_int8.tflite",
    "tflite_dir":      "exports/w600k_mbf_tflite",
    "input_shape":     (1, 3, 112, 112),
    "embedding_dim":   512,
    "max_size_mb":     6.0,    # budget for this model (leaves room for MediaPipe)
    "processed_dir":   "data/processed",
    "calib_samples":   200,
    # Android / iOS asset destinations
    "android_assets":  "../../DatalakeAttendance/android/app/src/main/assets",
    "ios_assets":      "../../DatalakeAttendance/ios/DatalakeAttendance",
}

# ─── Step 1: Download via insightface ────────────────────────────────────────

def download_insightface_onnx() -> str:
    """Download buffalo_sc and return absolute path to w600k_mbf.onnx."""
    try:
        import insightface
    except ImportError:
        print("insightface not found. Installing...")
        os.system(f"{sys.executable} -m pip install insightface")
        import insightface

    print(f"InsightFace version: {insightface.__version__}")

    from insightface.app import FaceAnalysis

    print(f"\nDownloading '{CONFIG['model_name']}' model pack...")
    print("(Downloads to ~/.insightface/models/ on first run — ~30 MB total)\n")

    app = FaceAnalysis(
        name      = CONFIG["model_name"],
        providers = ["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    # Locate the face recognition ONNX inside the downloaded pack
    home     = Path.home()
    onnx_dir = home / ".insightface" / "models" / CONFIG["model_name"]
    onnx_path = onnx_dir / CONFIG["onnx_filename"]

    if not onnx_path.exists():
        # Try alternative paths insightface might use
        candidates = list(onnx_dir.glob("*.onnx"))
        print(f"Files in model dir: {candidates}")
        # Pick the largest ONNX (recognition model is bigger than detection)
        rec_candidates = [f for f in candidates if "mbf" in f.name or "rec" in f.name]
        if rec_candidates:
            onnx_path = rec_candidates[0]
        elif candidates:
            onnx_path = max(candidates, key=lambda f: f.stat().st_size)
        else:
            raise FileNotFoundError(
                f"Could not find recognition ONNX in {onnx_dir}\n"
                "Files present: " + str(list(onnx_dir.iterdir()))
            )

    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"[OK] Found: {onnx_path}  ({size_mb:.2f} MB)")
    return str(onnx_path)


# ─── Step 2: Copy + Validate ONNX ────────────────────────────────────────────

def validate_and_copy_onnx(src_path: str, dst_path: str = CONFIG["onnx_out"]) -> str:
    """Validate the ONNX graph and copy to exports/."""
    import onnx

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(src_path)
    onnx.checker.check_model(model)
    print(f"ONNX graph validated OK")
    print(f"  Inputs:  {[i.name for i in model.graph.input]}")
    print(f"  Outputs: {[o.name for o in model.graph.output]}")

    # Optional graph simplification (pip install onnxsim)
    try:
        import onnxsim

        model_sim, ok = onnxsim.simplify(model)
        if ok:
            model = model_sim
            print("  Graph simplified [OK]")
        else:
            print("  Graph simplify skipped (onnxsim returned not ok)")
    except ImportError:
        print("  onnxsim not installed — copying graph without simplify")
        print("  (optional: pip install onnxsim)")

    onnx.save(model, dst_path)
    size_mb = os.path.getsize(dst_path) / 1024 / 1024
    print(f"[OK] ONNX -> {dst_path}  ({size_mb:.2f} MB)")
    return dst_path


# ─── Step 3: INT8 Quantization ────────────────────────────────────────────────

def _calibration_data(
    processed_dir: str = CONFIG["processed_dir"],
    n:             int  = CONFIG["calib_samples"],
    shape:         tuple = CONFIG["input_shape"],
) -> list:
    """Build calibration tensors from processed images or random noise."""
    import cv2
    import random

    path     = Path(processed_dir)
    imgs     = list(path.rglob("*.jpg")) + list(path.rglob("*.png"))
    mean     = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std      = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    _, _, H, W = shape
    tensors  = []

    if not imgs:
        print(f"  No images in {processed_dir} — using random noise for calibration.")
        return [np.random.randn(1, 3, H, W).astype(np.float32) for _ in range(min(n, 50))]

    selected = random.sample(imgs, min(n, len(imgs)))
    for p in selected:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H)).astype(np.float32) / 255.0
        img = (img - mean) / std
        tensors.append(np.transpose(img, (2, 0, 1))[np.newaxis])
    return tensors


def quantize_int8(
    onnx_path:      str = CONFIG["onnx_out"],
    onnx_int8_path: str = CONFIG["onnx_int8_out"],
) -> str:
    """Static INT8 quantization via onnxruntime."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    Path(onnx_int8_path).parent.mkdir(parents=True, exist_ok=True)
    calib = _calibration_data()

    # Detect the actual input name from the ONNX model
    import onnxruntime as ort
    sess      = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"  Model input name: '{input_name}'")

    class _Reader(CalibrationDataReader):
        def __init__(self, data, name):
            self._data  = data
            self._name  = name
            self._idx   = 0
        def get_next(self):
            if self._idx >= len(self._data):
                return None
            item = {self._name: self._data[self._idx]}
            self._idx += 1
            return item

    quantize_static(
        model_input             = onnx_path,
        model_output            = onnx_int8_path,
        calibration_data_reader = _Reader(calib, input_name),
        quant_format            = QuantFormat.QDQ,
        activation_type         = QuantType.QInt8,
        weight_type             = QuantType.QInt8,
        per_channel             = False,
        reduce_range            = False,
    )

    size_mb = os.path.getsize(onnx_int8_path) / 1024 / 1024
    print(f"[OK] INT8 ONNX -> {onnx_int8_path}  ({size_mb:.2f} MB)")
    return onnx_int8_path


# ─── Step 4: TFLite Conversion ────────────────────────────────────────────────

def convert_to_tflite(
    onnx_int8_path: str = CONFIG["onnx_int8_out"],
    tflite_dir:     str = CONFIG["tflite_dir"],
    tflite_path:    str = CONFIG["tflite_out"],
) -> str | None:
    """Convert INT8 ONNX to TFLite via onnx2tf (requires nhai-export env)."""
    try:
        import onnx2tf
    except ImportError:
        print("\n[SKIP] onnx2tf not installed in this Python environment.")
        print("  Training only needs: exports/w600k_mbf.onnx  (steps 1-3 above)")
        print("  For TFLite + deploy, use the export conda env:")
        print("    conda activate nhai-export")
        print("    pip install -r requirements-export.txt")
        print("    python src/get_pretrained.py --tflite-only")
        print("  Or after fine-tuning:")
        print("    python src/export_finetuned.py --step tflite")
        return None

    Path(tflite_dir).mkdir(parents=True, exist_ok=True)

    # onnx2tf needs a valid calibration input in cwd
    calib_fname = "calibration_image_sample_data_20x128x128x3_float32.npy"
    if not Path(calib_fname).exists():
        data = np.random.rand(20, 128, 128, 3).astype(np.float32)
        np.save(calib_fname, data, allow_pickle=False)

    onnx2tf.convert(
        input_onnx_file_path            = onnx_int8_path,
        output_folder_path              = tflite_dir,
        non_verbose                     = True,
        enable_batchmatmul_unfold       = True,
        disable_group_convolution       = True,
        copy_onnx_input_output_names_to_tflite = True,
        custom_input_op_name_np_data_path = [["input", str(Path(tflite_dir) / "_calib.npy")]],
    )

    generated = list(Path(tflite_dir).glob("*.tflite"))
    if not generated:
        raise FileNotFoundError(f"onnx2tf did not produce a .tflite in {tflite_dir}")

    Path(tflite_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(generated[0], tflite_path)

    size_mb = os.path.getsize(tflite_path) / 1024 / 1024
    print(f"[OK] TFLite -> {tflite_path}  ({size_mb:.2f} MB)")
    return tflite_path


# ─── Step 5: Size Validation + Deploy ────────────────────────────────────────

def validate_and_deploy(tflite_path: str = CONFIG["tflite_out"]) -> None:
    """Check size and copy to Android/iOS assets."""
    size_mb  = os.path.getsize(tflite_path) / 1024 / 1024
    max_mb   = CONFIG["max_size_mb"]
    mediapipe_mb = 3.2
    total_mb = size_mb + mediapipe_mb

    print(f"\n{'='*55}")
    print(f"  MobileFaceNet (InsightFace w600k_mbf)")
    print(f"  {size_mb:.2f} MB  +  MediaPipe {mediapipe_mb} MB  =  {total_mb:.2f} MB total")
    print(f"  Hackathon limit: 20 MB  ->  {'[OK] PASS' if total_mb < 20 else '[FAIL] EXCEEDS'}")
    print(f"{'='*55}\n")

    # Deploy to Android
    android_assets = Path(CONFIG["android_assets"])
    if android_assets.exists():
        dst = android_assets / "mobilefacenet_int8.tflite"
        shutil.copy(tflite_path, dst)
        print(f"[OK] Deployed to Android: {dst}")

    # Deploy to iOS
    ios_assets = Path(CONFIG["ios_assets"])
    if ios_assets.exists():
        dst = ios_assets / "mobilefacenet_int8.tflite"
        shutil.copy(tflite_path, dst)
        print(f"[OK] Deployed to iOS:     {dst}")

    if not android_assets.exists() and not ios_assets.exists():
        print("App directories not found relative to model-training/.")
        print(f"Manually copy:\n  {tflite_path}")
        print(f"  -> DatalakeAttendance/android/app/src/main/assets/mobilefacenet_int8.tflite")

    print(f"\n  IMPORTANT: Update FACE_VECTOR_SIZE in native code from 128 -> 512")
    print(f"  Files to update:")
    print(f"    src/native/FaceProcessor.ts        : FACE_VECTOR_SIZE = 512")
    print(f"    android/.../FaceAnalyzerModule.kt  : FACE_VECTOR_SIZE = 512")
    print(f"    ios/.../FaceAnalyzerModule.swift   : FACE_VECTOR_SIZE = 512")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run pipeline: download -> validate -> quantize -> (optional) tflite -> deploy."""
    import argparse

    parser = argparse.ArgumentParser(description="InsightFace w600k_mbf -> ONNX / TFLite")
    parser.add_argument(
        "--tflite-only",
        action="store_true",
        help="Only run TFLite conversion + deploy (needs onnx2tf / nhai-export env)",
    )
    parser.add_argument(
        "--onnx-only",
        action="store_true",
        help="Stop after ONNX export (for training on main Python env)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  InsightFace w600k_mbf Pre-Trained Model Pipeline")
    print("  WebFace600K -> ONNX -> INT8 -> TFLite")
    print("=" * 55 + "\n")

    if args.tflite_only:
        tflite = convert_to_tflite(
            CONFIG["onnx_int8_out"], CONFIG["tflite_dir"], CONFIG["tflite_out"]
        )
        if tflite:
            validate_and_deploy(tflite)
        return

    # 1. Download
    src = download_insightface_onnx()

    # 2. Validate + copy
    validate_and_copy_onnx(src, CONFIG["onnx_out"])

    # 3. INT8 quantize
    quantize_int8(CONFIG["onnx_out"], CONFIG["onnx_int8_out"])

    if args.onnx_only:
        print("\n[OK] ONNX ready for training: exports/w600k_mbf.onnx")
        print("  Next: python -u src/align_and_precompute.py")
        return

    # 4. TFLite (optional — onnx2tf usually not in training env)
    tflite = convert_to_tflite(
        CONFIG["onnx_int8_out"], CONFIG["tflite_dir"], CONFIG["tflite_out"]
    )

    # 5. Deploy if TFLite was produced or already on disk
    tflite_path = tflite or CONFIG["tflite_out"]
    if tflite_path and Path(tflite_path).exists():
        validate_and_deploy(tflite_path)
    else:
        print("\n[OK] Backbone ready for fine-tuning (no TFLite deploy this run).")
        print(f"  Use: exports/w600k_mbf.onnx")
        print("  App already has mobilefacenet_int8.tflite unless you re-export after training.")


if __name__ == "__main__":
    main()
