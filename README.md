# NHAI Hackathon 7.0 — Datalake Attendance

Offline-first mobile attendance for NHAI field teams: on-device face recognition, liveness check-in, encrypted local storage, GPS tagging, and batch cloud sync when connectivity returns.

**Repository:** [github.com/Immortal100100/NHAI-Datalake-Attendance](https://github.com/Immortal100100/NHAI-Datalake-Attendance)

---

## What is in this repository

| Included | Description |
|----------|-------------|
| **Source code** | React Native app (`DatalakeAttendance/`) + model training pipeline (`model-training/`) |
| **Deployed model weights** | `mobilefacenet_int8.tflite`, `face_landmarker.task` under `android/app/src/main/assets/` |
| **Fine-tuned checkpoint** | `model-training/checkpoints/finetuned_head.pt` |
| **Documentation** | This README only |

| Not included (by design) | Where to find it |
|-------------------------|-----------------|
| **Release APK** | Submitted separately to the hackathon portal — not stored in git (build locally; see [Building the app](#building-the-app)) |
| **Slide deck / PDF report** | Submitted separately — not in this repo |
| **Training dataset** | ~19 GB — download and prepare locally (see [Training data](#training-data-not-in-repo)) |
| **`node_modules` / Gradle build output** | Regenerated after clone (`npm install`, `gradlew`) |

---

## Highlights

- Fully **offline** enrolment and check-in; sync runs when network is available  
- On-device AI: **~10.4 MB** combined (face landmarker + fine-tuned MobileFaceNet FP16 TFLite)  
- **128-D** face embeddings, cosine matching, L2 normalization in native Kotlin  
- **Liveness:** head-movement gate over 5 frames (EAR/MAR landmarks extensible)  
- **Storage:** MMKV with AES-256 encryption  
- **GPS** on each check-in (4 s timeout, 60 s cache fallback)  
- **Sync:** JSON batch POST; demo endpoint on webhook.site (swap for API Gateway in production)

**Platforms:** Android 8.0+ built and tested on device. **iOS 12+:** React Native UI + Swift module scaffold present; release build and on-device testing **not** completed for this submission.

---

## Problem statement (Hackathon 7.0)

> How can we accurately and securely authenticate field personnel using facial recognition and liveness detection on standard mid-range mobile devices without any active internet connection, while ensuring the AI model remains lightweight and seamlessly integrates with a React Native application on both Android and iOS devices?

| Constraint | Requirement | This project |
|------------|-------------|--------------|
| Framework | React Native (Android + iOS) | RN 0.85 · TypeScript |
| Model size | ~20 MB (smaller is better) | **~10.4 MB** on device |
| Speed | < 1 s recognition | **~100–500 ms** per inference; ~2–5 s full check-in (5-frame liveness) |
| Accuracy | > 95% | **80.6% TAR@FAR=1%** — see [Accuracy](#accuracy) |
| Liveness | Blink / smile / head turn | Head-turn **active**; EAR/MAR ready |
| Sync + purge | AWS when online | Batch POST + purge on HTTP 2xx |

---

## Repository structure

```
NHAI-Datalake-Attendance/
├── README.md
├── DatalakeAttendance/                 # React Native 0.85 app
│   ├── src/
│   │   ├── screens/                    # Register, CheckIn, AdminSync
│   │   ├── native/FaceProcessor.ts     # thresholds, matching, liveness
│   │   ├── db/mmkv.ts                  # encrypted profiles + attendance
│   │   ├── config/syncConfig.ts        # sync URL & API key
│   │   ├── utils/attendanceSync.ts
│   │   └── components/SyncBootstrap.tsx
│   ├── android/app/src/main/
│   │   ├── assets/                     # bundled TFLite models
│   │   └── java/.../faceprocessor/     # FaceAnalyzerModule.kt
│   ├── ios/                            # scaffold (not device-tested)
│   └── local-sync-server/              # optional LAN test server
└── model-training/
    ├── src/                            # align, finetune, export scripts
    ├── checkpoints/finetuned_head.pt
    └── requirements.txt
```

---

## How it works

### 1. Offline enrolment

User enters name and employee code → camera captures frames → native module outputs 128-D embedding → averaged and stored in encrypted MMKV.

### 2. Liveness check-in

Five frames captured → `evaluateLiveness()` requires head movement → cosine match against gallery (threshold **0.45**, top1–top2 margin **0.08**) → attendance record with GPS saved locally (`synced: false`).

### 3. Auto sync and purge

`SyncBootstrap` listens for connectivity → unsynced records batched as JSON → `POST` to configured endpoint → on HTTP **2xx**, records marked synced and purged from device.

### Architecture

```
Camera (Vision Camera)
    → face_landmarker.task + mobilefacenet_int8.tflite (Kotlin)
    → evaluateLiveness() → matchAgainstGallery()
    → MMKV (profiles + attendance queue)

When online → attendanceSync.ts → POST → cloud endpoint → purge
```

| Component | Location |
|-----------|----------|
| UI | `DatalakeAttendance/src/screens/` |
| Face / match / liveness | `DatalakeAttendance/src/native/FaceProcessor.ts` |
| Encrypted storage | `DatalakeAttendance/src/db/mmkv.ts` |
| Sync | `DatalakeAttendance/src/utils/attendanceSync.ts` |
| Native inference | `DatalakeAttendance/android/app/src/main/java/.../FaceAnalyzerModule.kt` |

---

## On-device models (in repo)

| Asset | Path | Size (approx.) |
|-------|------|----------------|
| Recognition | `DatalakeAttendance/android/app/src/main/assets/mobilefacenet_int8.tflite` | ~7.2 MB |
| Face mesh | `DatalakeAttendance/android/app/src/main/assets/face_landmarker.task` | ~0.8 MB |

L2 normalization runs in Kotlin after TFLite inference (not inside the graph).

---

## Accuracy

| Metric | Value |
|--------|-------|
| **TAR @ FAR = 1%** | **80.6%** (best reproducible run, held-out validation) |
| Training corpus | ~41k images, 362 identities (`all_indian` — not in repo) |
| PS target | 95% |

Validation is below the problem-statement target because training used a limited public corpus on a single GPU with a frozen backbone. The same export pipeline can scale with a larger NHAI-provided dataset and multi-GPU training without changing the app architecture.

---

## Liveness detection

| Method | Status | Implementation |
|--------|--------|----------------|
| **Head turn** | Active | `evaluateLiveness()` in `FaceProcessor.ts` — face box motion over 5 frames |
| **Blink (EAR)** | Extensible | Landmarks in native module; threshold in `faceMath.ts` |
| **Smile (MAR)** | Extensible | Mouth landmarks; enable via config |

Rigid photos or screens fail the motion gate.

---

## Cloud sync

Configuration: [`DatalakeAttendance/src/config/syncConfig.ts`](DatalakeAttendance/src/config/syncConfig.ts)

| Setting | Demo value |
|---------|------------|
| `AWS_SYNC_URL` | `https://webhook.site/5678eed5-7875-466a-a724-dd3761243de4` |
| `timeoutMs` | `15000` |
| Demo inbox | [webhook.site/#!/5678eed5-7875-466a-a724-dd3761243de4](https://webhook.site/#!/5678eed5-7875-466a-a724-dd3761243de4) |

**Payload** uploads attendance metadata only (no face embeddings):

```json
{
  "records": [{
    "id": "uuid",
    "userId": "profile-id",
    "employeeCode": "NHAI-001",
    "name": "Field Worker",
    "checkInTime": 1717061234567,
    "latitude": 28.6139,
    "longitude": 77.2090,
    "locationAccuracy": 12.5,
    "livenessPassed": true,
    "similarityScore": 0.583
  }],
  "syncedAt": "2026-05-30T06:00:00.000Z",
  "source": "datalake-attendance-mobile"
}
```

For production: set `AWS_SYNC_URL` to your API Gateway URL and `SYNC_CONFIG.apiKey` to a Bearer token, then rebuild the APK.

---

## Getting started (clone and run)

### Prerequisites

- **Node.js** 18+
- **JDK** 17
- **Android SDK** API 26+
- **Android device or emulator** with camera (physical device recommended for demo)

### 1. Clone

```bash
git clone https://github.com/Immortal100100/NHAI-Datalake-Attendance.git
cd NHAI-Datalake-Attendance/DatalakeAttendance
npm install
```

### 2. Run debug build (development)

```bash
npm start
# separate terminal:
npm run android
```

### 3. Build release APK (for device install)

The **release APK is not committed** to this repository. Build it locally:

```bash
cd android
./gradlew assembleRelease
```

**Output (local only, not in git):**

`android/app/build/outputs/apk/release/app-release.apk`

Install on a connected device:

```bash
adb install -r android/app/build/outputs/apk/release/app-release.apk
```

Submit the APK file separately per hackathon instructions.

---

## Optional: local sync test server

```bash
cd DatalakeAttendance/local-sync-server
npm install
npm start
```

Set in `syncConfig.ts`:

```typescript
export const USE_LOCAL_SYNC_SERVER = true;
export const LOCAL_SYNC_HOST = 'http://<your-lan-ip>:8787/attendance';
```

Rebuild the APK after changing sync settings.

---

## Training pipeline (optional)

Requires Python 3.10+, CUDA GPU recommended, and a prepared dataset under `model-training/data/` (not in repo).

```bash
cd model-training
pip install -r requirements.txt

# 1. Prepare embeddings (after dataset + manifest exist)
python -u src/align_and_precompute.py \
  --manifest_path data/all_indian/manifest.csv \
  --processed_dir data/all_indian \
  --cache_path data/all_indian/embed_cache_aligned.npz \
  --n_aug 10

# 2. Fine-tune projection head
python -u src/finetune.py \
  --embed_cache data/all_indian/embed_cache_aligned.npz \
  --epochs 60 --lr 5e-2

# 3. Export and deploy to app assets
python src/export_finetuned.py --step fuse --backbone exports/w600k_mbf.onnx
python src/export_finetuned.py --step quantize
# TFLite step needs conda env from requirements-export.txt
python src/export_finetuned.py --step tflite
python src/export_finetuned.py --step deploy
```

Then rebuild the Android app. Large ONNX export artifacts are gitignored under `model-training/exports/`.

---

## Training data (not in repo)

Training images are **not** versioned (multi-gigabyte). To train from scratch:

1. Obtain Indian face data (e.g. IMFDB, Kaggle Bollywood, or custom identity folders).  
2. Place under `model-training/data/` and build a manifest (`src/build_all_indian_manifest.py`, `src/dataset_indian.py`).  
3. Minimum **5+ images per identity** recommended.

IMFDB requires academic registration: [cvit.iiit.ac.in/projects/IMFDB](http://cvit.iiit.ac.in/projects/IMFDB/).

---

## Integrating into Datalake 3.0

| Step | Action |
|------|--------|
| 1 | Copy `DatalakeAttendance/src/` into the host app (e.g. `src/features/attendance/`) |
| 2 | Copy `android/.../faceprocessor/` and asset files into the host Android project |
| 3 | Register `FaceAnalyzerPackage` in `MainApplication.kt` |
| 4 | Add dependencies from `DatalakeAttendance/package.json` |
| 5 | Set production `AWS_SYNC_URL` in `syncConfig.ts` |
| 6 | Mount `<SyncBootstrap />` at the app root |

---

## Hackathon submission mapping

| Deliverable | Provided via |
|-------------|--------------|
| Source code | **This GitHub repository** |
| Working Android prototype (APK) | **Separate upload** — build with `gradlew assembleRelease` |
| Presentation / technical PDF | **Separate upload** — not in repo |
| Offline liveness | Head-turn gate in `FaceProcessor.ts` |
| AWS sync + purge | `attendanceSync.ts` + demo webhook POST |
| Open-source stack | React Native, TFLite, InsightFace, MediaPipe, MMKV, etc. |
| Model ≤ 20 MB | ~10.4 MB bundled assets |

### Evaluation criteria (100 marks)

| Criterion | Marks | Evidence in repo |
|-----------|------:|------------------|
| Innovation | 30 | Edge TFLite stack, compression pipeline, offline liveness |
| Feasibility | 30 | RN integration, mid-range device testing, <1 s inference class |
| Scalability | 20 | Sync/purge flow, Indian-domain fine-tune, accuracy improvement path |
| Presentation | 20 | README + source clarity; deck/PDF submitted separately |

---

## Demo flow (~5 minutes)

1. **Enrol** — Register tab → name + employee code → capture face (offline).  
2. **Check-in** — Slow head turn → match with similarity score.  
3. **Negative test** — Different person → rejected.  
4. **Offline** — Airplane mode → check-in still succeeds; pending count increases.  
5. **Sync** — Restore network → Sync tab or auto-sync → verify JSON at webhook inbox.

---

## Known limitations

- **80.6% TAR@FAR=1%** below the 95% PS benchmark — improvable with NHAI-scale data and GPUs.  
- **iOS** not built or tested on device for this submission.  
- **webhook.site** is for demo only; use API Gateway + auth in production.  
- **Check-in UX** uses 5 frames (~2–5 s) to satisfy liveness requirements by design.

---

## Tech stack

React Native 0.85 · TypeScript · Vision Camera · react-native-mmkv · NetInfo · Geolocation · Kotlin · TensorFlow Lite · PyTorch · InsightFace · ArcFace · ONNX · onnx2tf

---

## License and data

Application code and training scripts in this repository use open-source dependencies (MIT / Apache 2.0). Third-party training datasets (e.g. IMFDB) may require separate registration or licenses from their providers.
