# NHAI Hackathon — Datalake Attendance

Offline-first React Native attendance for NHAI field teams: on-device face enrolment, liveness check-in, encrypted storage, GPS tagging, and batch cloud sync.

| Deliverable | Path |
|-------------|------|
| **Release APK** | Build → `DatalakeAttendance/android/app/build/outputs/apk/release/app-release.apk` |
| **On-device model** | `DatalakeAttendance/android/app/src/main/assets/mobilefacenet_int8.tflite` |

**Platform:** Android built and tested on device. **iOS:** Swift scaffold only — not built or device-tested.

---

## Problem statement

> How can we accurately and securely authenticate field personnel using facial recognition and liveness detection on standard mid-range mobile devices without any active internet connection, while ensuring the AI model remains lightweight and seamlessly integrates with a React Native application on both Android and iOS devices?

| Constraint | Requirement | Achieved |
|------------|-------------|----------|
| Framework | React Native (Android + iOS) | RN 0.85; Android verified |
| Model size | ~20 MB | **~10.4 MB** on device |
| Speed | < 1 s recognition | **~100–500 ms** per inference; ~2–5 s full check-in (5-frame liveness) |
| Accuracy | > 95% | **80.6% TAR@FAR=1%** — improvable with NHAI dataset + GPUs |
| Liveness | Blink / smile / head turn | **Head-turn active**; EAR/MAR ready |
| Sync + purge | AWS when online | Batch POST + purge on HTTP 2xx (verified) |

---

## Repository layout

```
NHAI/
├── README.md                    ← this file (only documentation)
├── hackathon_doc7.pdf           ← technical report
├── DatalakeAttendance/          ← React Native app
│   ├── src/screens/             Register, CheckIn, AdminSync
│   ├── src/native/FaceProcessor.ts
│   ├── src/db/mmkv.ts
│   ├── src/config/syncConfig.ts
│   └── android/.../faceprocessor/FaceAnalyzerModule.kt
└── model-training/              ← fine-tune + export pipeline
    ├── src/                     align, finetune, export scripts
    └── checkpoints/finetuned_head.pt
```

**Dataset is not in the repo** (~19 GB). Delete locally with the command in [Dataset](#dataset-not-in-repo) below.

---

## Solution overview

1. **Offline enrolment** — capture face → 128-D embedding → MMKV AES-256  
2. **Liveness check-in** — head-turn over 5 frames → cosine match (≥ 0.45) → GPS-tagged record  
3. **Encrypted storage** — profiles + attendance queue on device  
4. **Auto sync** — NetInfo → batch POST → mark synced → purge on HTTP 2xx  

---

## Architecture

```
Camera → MediaPipe Face Mesh (~3.2 MB) → landmarks + face box
      → MobileFaceNet FP16 (~7.2 MB) → 128-D → L2 norm (Kotlin)
      → evaluateLiveness() → matchAgainstGallery() → MMKV record

When online: SyncBootstrap → POST JSON → AWS / webhook → purge
```

| Layer | Path |
|-------|------|
| UI | `DatalakeAttendance/src/screens/` |
| Face logic | `src/native/FaceProcessor.ts` |
| Storage | `src/db/mmkv.ts` (AES-256) |
| Sync | `src/utils/attendanceSync.ts` |
| Auto-sync | `src/components/SyncBootstrap.tsx` |
| Native | `android/.../FaceAnalyzerModule.kt` |

**Assets:** `android/app/src/main/assets/mobilefacenet_int8.tflite` (7.2 MB)

---

## AI model

| Property | Value |
|----------|-------|
| Backbone | InsightFace `w600k_mbf` (frozen) |
| Head | ArcFace MLP 512→256→128 |
| TAR@FAR=1% | **80.6%** (epoch 45, `all_indian` held-out) |
| Threshold | cosine ≥ **0.45** |
| Liveness frames | **5** (`VERIFICATION_WINDOW_FRAMES`) |

**Accuracy note:** Below the 95% PS target because training used ~41k images / 362 IDs on one consumer GPU with a frozen backbone. Same export pipeline scales with NHAI data + multi-GPU — no app rewrite.

### Retrain & export

```powershell
cd model-training
$env:PYTHONUTF8 = "1"

# After placing dataset under data/all_indian/ (see Dataset section)
python -u src/align_and_precompute.py `
  --manifest_path data/all_indian/manifest.csv `
  --processed_dir data/all_indian `
  --cache_path data/all_indian/embed_cache_aligned.npz `
  --n_aug 10

python -u src/finetune.py --embed_cache data/all_indian/embed_cache_aligned.npz --epochs 60 --lr 5e-2

python src/export_finetuned.py --step fuse --backbone exports/w600k_mbf.onnx
python src/export_finetuned.py --step quantize
conda run -n nhai-export python src/export_finetuned.py --step tflite
python src/export_finetuned.py --step deploy
# Rebuild APK
```

---

## Dataset (not in repo)

Training data (`all_indian`, ~41k images, 362 identities) is **excluded from git**. To remove it locally, run:

```powershell
# Delete entire dataset folder (~19 GB) — run from any terminal
cmd /c "rmdir /s /q \"C:\Users\kunal\Desktop\NHAI\model-training\data\""
```

Or delete in stages (if the full delete hangs):

```powershell
cmd /c "rmdir /s /q \"C:\Users\kunal\Desktop\NHAI\model-training\data\downloads\""
cmd /c "rmdir /s /q \"C:\Users\kunal\Desktop\NHAI\model-training\data\all_indian\""
cmd /c "rmdir /s /q \"C:\Users\kunal\Desktop\NHAI\model-training\data\""
```

**To re-download for training:** IMFDB (IIIT Hyderabad), Kaggle Bollywood, or custom folders under `model-training/data/raw/`. Minimum 5+ images per identity. Use `src/dataset_indian.py` and `src/build_all_indian_manifest.py`.

---

## Liveness detection

| Method | Status | Detail |
|--------|--------|--------|
| **Head turn** | Active | `evaluateLiveness()` — face box X/Y/width range over 5 frames; threshold 0.06 X-range or 0.08 total movement |
| **EAR blink** | Ready | MediaPipe eye landmarks; EAR < 0.25 in `faceMath.ts` |
| **MAR smile** | Ready | Mouth landmarks; MAR > 0.6 |

Photos/screens fail because the face box stays rigid across frames.

---

## Sync API

**Config:** `DatalakeAttendance/src/config/syncConfig.ts`

| Setting | Demo value |
|---------|------------|
| `AWS_SYNC_URL` | `https://webhook.site/5678eed5-7875-466a-a724-dd3761243de4` |
| `timeoutMs` | 15000 |
| View inbox | https://webhook.site/#!/5678eed5-7875-466a-a724-dd3761243de4 |

**POST body (metadata only — no face embeddings):**

```json
{
  "records": [{
    "id": "...", "userId": "...", "employeeCode": "NHAI-001",
    "name": "...", "checkInTime": 1717061234567,
    "latitude": 28.6139, "longitude": 77.2090,
    "livenessPassed": true, "similarityScore": 0.583
  }],
  "syncedAt": "2026-05-30T06:00:00.000Z",
  "source": "datalake-attendance-mobile"
}
```

On HTTP **2xx**: `markRecordsSynced()` → `purgeSyncedRecords()`.

**Verified demo:** `livenessPassed: true`, `similarityScore: 0.583`, GPS ~27.87°N 78.08°E (webhook.site POST).

**Production:** Replace URL with API Gateway; set `SYNC_CONFIG.apiKey` Bearer token.

---

## Build & deploy

### Prerequisites

Node 18+, Android SDK API 26+, JDK 17, USB/wireless ADB.

### Release APK

```powershell
cd DatalakeAttendance
npm install

cd android
.\gradlew assembleRelease

$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
& $adb install -r app\build\outputs\apk\release\app-release.apk
```

### Sync endpoint options

```typescript
// syncConfig.ts — Option A: demo webhook (current)
export const USE_LOCAL_SYNC_SERVER = false;
export const AWS_SYNC_URL = 'https://webhook.site/<uuid>';

// Option B: LAN test server
cd local-sync-server && npm start
export const USE_LOCAL_SYNC_SERVER = true;
export const LOCAL_SYNC_HOST = 'http://192.168.x.x:8787/attendance';

// Option C: production
export const AWS_SYNC_URL = 'https://<api-id>.execute-api.<region>.amazonaws.com/prod/attendance';
```

Rebuild APK after changing config.

---

## Demo script (~5 min)

1. **Enrol** — Register tab → name + employee code → capture face (offline)  
2. **Check-in** — Head turn → green match with similarity score  
3. **Negative test** — Different person → no match  
4. **Offline** — Airplane mode → check-in still works → pending count rises  
5. **Sync** — Online → Sync tab → show webhook JSON POST  

**Talking points:** Offline AI (~10 MB), 80.6% TAR (scalable), encrypted MMKV, batch sync with purge.

---

## Datalake 3.0 integration

| Step | Action |
|------|--------|
| 1 | Copy `DatalakeAttendance/src/` → host `src/features/attendance/` |
| 2 | Copy `android/.../faceprocessor/` + TFLite assets |
| 3 | Register `FaceAnalyzerPackage` in `MainApplication.kt` |
| 4 | Add npm deps: vision-camera, mmkv, netinfo, geolocation |
| 5 | Set production `AWS_SYNC_URL` in `syncConfig.ts` |
| 6 | Mount `<SyncBootstrap />` at app root |

---

## Evaluation criteria (100 marks)

| Criterion | Marks | Evidence |
|-----------|------:|----------|
| Innovation | 30 | ~10.4 MB edge AI, compression pipeline, offline liveness |
| Feasibility | 30 | RN 0.85 integration, <1 s inference class, Android device test |
| Scalability | 20 | Sync/purge verified, Indian fine-tune, accuracy scale path |
| Presentation | 20 | PPT, PDF report, this README, source code |

---

## Deliverables checklist

| Item | Status | Location |
|------|--------|----------|
| Source code | Done | This repo |
| Android APK | Done | Build from `android/` |
| iOS | Partial | Scaffold only |
| Offline liveness | Done | Head-turn gate |
| AWS sync + purge | Done | webhook proof screenshot |
| Open-source only | Done | MIT/Apache stack |
| Model ≤ 20 MB | Done | ~10.4 MB |

---

## Clean repo before git push

```powershell
# Dataset (you run this)
cmd /c "rmdir /s /q \"C:\Users\kunal\Desktop\NHAI\model-training\data\""

# Build artifacts & dependencies (regenerate with npm install / gradlew)
cmd /c "rmdir /s /q \"C:\Users\kunal\Desktop\NHAI\DatalakeAttendance\node_modules\""
cmd /c "rmdir /s /q \"C:\Users\kunal\Desktop\NHAI\DatalakeAttendance\android\app\build\""
```

---

## Known limitations

- **80.6% TAR** below 95% PS target — needs larger NHAI dataset + multi-GPU  
- **iOS** not built or tested  
- **webhook.site** is demo-only — replace for production  
- **5-frame check-in** ~2–5 s by design (liveness requirement)

---

## Tech stack

React Native 0.85 · Vision Camera v4 · MMKV · NetInfo · Geolocation · Kotlin TFLite · PyTorch · InsightFace · ArcFace · ONNX · onnx2tf

## License

Open-source components only (MIT / Apache 2.0). IMFDB requires academic registration for dataset download.
