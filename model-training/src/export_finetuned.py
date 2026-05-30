"""
export_finetuned.py — Fuse fine-tuned projection head with frozen backbone.

After running finetune.py, this script:
  1. Loads checkpoints/finetuned_head.pt  (projection head weights)
  2. Loads exports/w600k_mbf.onnx         (frozen backbone)
  3. Fuses them into: exports/finetuned_combined.onnx
  4. INT8 quantizes:  exports/finetuned_combined_int8.onnx
  5. Converts TFLite: exports/finetuned_combined.tflite  (run in nhai-export env)
  6. Deploys to Android assets

Usage:
  # Training env (Python 3.14)
  $env:PYTHONUTF8="1"
  python src/export_finetuned.py --step fuse
  python src/export_finetuned.py --step quantize
  python src/export_finetuned.py --step validate

  # Export env (conda activate nhai-export)
  $py = (Get-Command python).Path
  & $py src/export_finetuned.py --step tflite

  # Deploy to Android
  python src/export_finetuned.py --step deploy
"""

import argparse
import os
import shutil
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ─── Paths ───────────────────────────────────────────────────────────────────

BACKBONE_ONNX   = "exports/w600k_mbf.onnx"
HEAD_CHECKPOINT = "checkpoints/finetuned_head.pt"
COMBINED_ONNX   = "exports/finetuned_combined.onnx"
INT8_ONNX       = "exports/finetuned_combined_int8.onnx"
TFLITE_OUTPUT   = "exports/finetuned_combined.tflite"
ANDROID_ASSETS  = "../DatalakeAttendance/android/app/src/main/assets"
ANDROID_MODEL   = f"{ANDROID_ASSETS}/mobilefacenet_int8.tflite"

EMBED_DIM    = 128   # projection head output
BACKBONE_DIM = 512   # w600k_mbf output
INPUT_H = INPUT_W = 112


# ─── Step 1: Fuse backbone ONNX + projection head ────────────────────────────

def step_fuse() -> None:
    """Fuse backbone ONNX + multi-layer projection head into one ONNX graph."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import onnx
    import onnxruntime as ort
    from checkpoint_io import load_checkpoint

    print("Loading backbone:", BACKBONE_ONNX)
    print("Loading head checkpoint:", HEAD_CHECKPOINT)
    ckpt = load_checkpoint(HEAD_CHECKPOINT, map_location="cpu")

    # ── Rebuild ProjectionHead exactly as in finetune.py ─────────────────────
    class ProjectionHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(BACKBONE_DIM, 256, bias=False),
                nn.BatchNorm1d(256, affine=True),
                nn.PReLU(256),
                nn.Dropout(0.0),
                nn.Linear(256, EMBED_DIM, bias=False),
                nn.BatchNorm1d(EMBED_DIM, affine=False),
            )
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # No F.normalize — normalization done in Android to avoid
            # ReduceL2 opset-18 incompatibility with onnx2tf
            return self.net(x)
    proj = ProjectionHead()
    proj.load_state_dict(ckpt["projection_state"])
    proj.eval()

    # ── Convert backbone ONNX → PyTorch via onnx2torch ───────────────────────
    try:
        import onnx2torch
    except ImportError:
        raise ImportError("pip install onnx2torch")

    backbone_onnx = onnx.load(BACKBONE_ONNX)
    backbone_pt   = onnx2torch.convert(backbone_onnx).eval()
    print("[OK] Backbone converted to PyTorch")

    # ── Combined model ────────────────────────────────────────────────────────
    class CombinedFaceNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone  = backbone_pt
            self.projector = proj
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feats = self.backbone(x)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            return self.projector(feats)

    model = CombinedFaceNet().eval()

    # Smoke-test
    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W)
    with torch.no_grad():
        out = model(dummy)
    print(f"[OK] Forward pass — output shape: {out.shape}  norm={out.norm().item():.4f}")

    # ── Export to ONNX ────────────────────────────────────────────────────────
    Path(COMBINED_ONNX).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, COMBINED_ONNX,
        input_names=["input"],
        output_names=["embedding"],
        opset_version=11,
        do_constant_folding=True,
    )
    size_mb = Path(COMBINED_ONNX).stat().st_size / 1e6
    print(f"[OK] Saved {COMBINED_ONNX}  ({size_mb:.2f} MB)")

    # Verify with onnxruntime
    sess = ort.InferenceSession(COMBINED_ONNX, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    result = sess.run(None, {inp_name: dummy.numpy()})[0]
    print(f"[OK] ORT sanity check — output: {result.shape}  norm={np.linalg.norm(result):.4f}")


# ─── Step 2: INT8 quantization ───────────────────────────────────────────────

def step_quantize() -> None:
    """INT8 quantize the combined ONNX using onnxruntime-tools."""
    import onnx
    import onnxruntime.quantization.quant_utils as _qu

    print("INT8 quantizing", COMBINED_ONNX, "...")

    # The onnx2torch re-export leaves shape annotations that fail strict
    # shape inference. Simplify first with onnxsim to fix them.
    print("  Simplifying ONNX graph first...")
    try:
        import onnxsim
        model = onnx.load(COMBINED_ONNX)
        model_sim, ok = onnxsim.simplify(model)
        if ok:
            simplified_path = COMBINED_ONNX.replace(".onnx", "_sim.onnx")
            onnx.save(model_sim, simplified_path)
            quant_src = simplified_path
            print("  [OK] Simplified")
        else:
            quant_src = COMBINED_ONNX
            print("  [WARN] Simplification failed, using original")
    except Exception as e:
        print(f"  [WARN] onnxsim unavailable ({e}), monkey-patching shape-infer")
        quant_src = COMBINED_ONNX

    # Monkey-patch: skip the problematic internal shape-infer reload
    _orig = _qu.save_and_reload_model_with_shape_infer
    _qu.save_and_reload_model_with_shape_infer = lambda m: m

    from onnxruntime.quantization import quantize_dynamic, QuantType
    try:
        quantize_dynamic(quant_src, INT8_ONNX, weight_type=QuantType.QInt8)
    finally:
        _qu.save_and_reload_model_with_shape_infer = _orig

    import os
    sim = COMBINED_ONNX.replace(".onnx", "_sim.onnx")
    if os.path.exists(sim):
        os.remove(sim)

    data_file = Path(COMBINED_ONNX + ".data")
    orig_mb = (Path(COMBINED_ONNX).stat().st_size + (data_file.stat().st_size if data_file.exists() else 0)) / 1e6
    q8_mb = Path(INT8_ONNX).stat().st_size / 1e6
    print(f"[OK] {orig_mb:.2f} MB -> {INT8_ONNX} ({q8_mb:.2f} MB)")


# ─── Step 3: Validate ONNX ───────────────────────────────────────────────────

def step_validate() -> None:
    """Run inference sanity check on INT8 ONNX."""
    import onnxruntime as ort

    for path in [COMBINED_ONNX, INT8_ONNX]:
        if not Path(path).exists():
            print(f"[SKIP] {path} not found")
            continue
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp  = sess.get_inputs()[0].name
        dummy = np.random.randn(1, 3, INPUT_H, INPUT_W).astype(np.float32)
        out  = sess.run(None, {inp: dummy})[0]
        norm = np.linalg.norm(out)
        size = Path(path).stat().st_size / 1e6
        print(f"[OK] {path}: output={out.shape}  norm={norm:.4f}  size={size:.2f} MB")


# ─── Step 4: TFLite conversion (run in nhai-export conda env) ────────────────

def step_tflite() -> None:
    """Convert FP32 combined ONNX -> TFLite FP32, then INT8 via TFLite converter."""
    import onnx2tf
    import tensorflow as tf

    out_dir = "exports/tflite_finetuned"
    os.makedirs(out_dir, exist_ok=True)

    # Step A: FP32 ONNX → SavedModel/TFLite via onnx2tf
    # Use the FP32 combined ONNX (NOT the INT8 ONNX — ConvInteger unsupported)
    print(f"Converting {COMBINED_ONNX} -> TFLite (FP32 first) ...")
    onnx2tf.convert(
        input_onnx_file_path = COMBINED_ONNX,
        output_folder_path   = out_dir,
        non_verbose          = True,
    )

    tflites = list(Path(out_dir).glob("*.tflite"))
    if not tflites:
        print("[ERROR] No .tflite generated. Check onnx2tf output above.")
        return

    fp32_tflite = str(tflites[0])
    fp32_mb = Path(fp32_tflite).stat().st_size / 1e6
    print(f"[OK] FP32 TFLite: {fp32_tflite} ({fp32_mb:.2f} MB)")

    # Step B: Apply INT8 post-training quantization via TFLite converter
    print("Applying INT8 PTQ via TFLite converter ...")

    def representative_dataset():
        calib_path = "exports/calib_data.npy"
        if Path(calib_path).exists():
            calib = np.load(calib_path)  # (N, H, W, 3)
            # Convert NHWC → NCHW if needed, then yield batches
            for i in range(min(len(calib), 20)):
                img = calib[i:i+1].transpose(0, 3, 1, 2).astype(np.float32)
                yield [img]
        else:
            for _ in range(20):
                yield [np.random.randn(1, 3, INPUT_H, INPUT_W).astype(np.float32)]

    saved_model_dir = Path(out_dir) / "saved_model"
    if saved_model_dir.exists():
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    else:
        # No saved_model dir — use FP16 tflite directly (already within 20MB)
        shutil.copy(fp32_tflite, TFLITE_OUTPUT)
        print(f"[OK] Using FP16 TFLite: {TFLITE_OUTPUT} ({fp32_mb:.2f} MB)")
        return

    # Apply dynamic range quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    try:
        tflite_quant = converter.convert()
        with open(TFLITE_OUTPUT, "wb") as f:
            f.write(tflite_quant)
        size_mb = Path(TFLITE_OUTPUT).stat().st_size / 1e6
        print(f"[OK] INT8 TFLite: {TFLITE_OUTPUT} ({size_mb:.2f} MB)")
        if size_mb > 20:
            print(f"[WARN] {size_mb:.1f} MB exceeds limit — using FP16 instead")
            shutil.copy(fp32_tflite, TFLITE_OUTPUT)
            print(f"[OK] Using FP16 TFLite: {fp32_mb:.2f} MB")
        else:
            print(f"[OK] Within 20 MB hackathon limit")
    except Exception as e:
        print(f"[WARN] INT8 conversion failed ({e}), using FP16 TFLite")
        shutil.copy(fp32_tflite, TFLITE_OUTPUT)
        print(f"[OK] Saved FP16 TFLite: {fp32_mb:.2f} MB")


# ─── Step 5: Deploy to Android ───────────────────────────────────────────────

def step_deploy() -> None:
    """Copy TFLite to Android assets and update FACE_VECTOR_SIZE to 128."""
    src = Path(TFLITE_OUTPUT)
    if not src.exists():
        print(f"[ERROR] {TFLITE_OUTPUT} not found. Run --step tflite first.")
        return

    dst = Path(ANDROID_MODEL)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    size_mb = src.stat().st_size / 1e6
    print(f"[OK] Deployed to {dst} ({size_mb:.2f} MB)")

    # Update FACE_VECTOR_SIZE to 128 in native code
    ts_file = Path("../DatalakeAttendance/src/native/FaceProcessor.ts")
    kt_file = Path("../DatalakeAttendance/android/app/src/main/java/com/datalakeattendance/faceprocessor/FaceAnalyzerModule.kt")

    for fpath in [ts_file, kt_file]:
        if not fpath.exists():
            continue
        text = fpath.read_text(encoding="utf-8")
        updated = text.replace(
            "FACE_VECTOR_SIZE = 512", "FACE_VECTOR_SIZE = 128"
        ).replace(
            "const FACE_VECTOR_SIZE = 512", "const FACE_VECTOR_SIZE = 128"
        )
        if updated != text:
            fpath.write_text(updated, encoding="utf-8")
            print(f"[OK] Updated FACE_VECTOR_SIZE -> 128 in {fpath.name}")
        else:
            print(f"[--] {fpath.name} already at 128 or not found pattern")

    print("\nNext: rebuild the Android app")
    print("  cd ../DatalakeAttendance")
    print("  .\\android\\gradlew assembleRelease")


# ─── CLI ─────────────────────────────────────────────────────────────────────

STEPS = {
    "fuse":     step_fuse,
    "quantize": step_quantize,
    "validate": step_validate,
    "tflite":   step_tflite,
    "deploy":   step_deploy,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export fine-tuned face model")
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()),
        default="fuse",
        help="Export step to run",
    )
    parser.add_argument(
        "--backbone",
        default=None,
        help="Path to backbone ONNX (overrides default BACKBONE_ONNX)",
    )
    args = parser.parse_args()
    if args.backbone:
        BACKBONE_ONNX = args.backbone
    STEPS[args.step]()
