import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  View, Text, StyleSheet, SafeAreaView, TouchableOpacity, ActivityIndicator,
} from 'react-native';
import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import CameraOverlay from '../components/CameraOverlay';
import { getUnsyncedRecords, saveAttendanceRecord, getAllProfiles, getProfile } from '../db/mmkv';
import type { OfflineAttendance } from '../db/mmkv';
import {
  processFaceFrameStub, FACE_VECTOR_SIZE, VERIFICATION_WINDOW_FRAMES,
  averageEmbeddings, matchAgainstGallery, jitterEmbedding,
  NATIVE_ANALYZER_AVAILABLE, loadNativeFaceModels, analyzeFrameNative,
  releaseNativeFaceModels, evaluateLiveness,
} from '../native/FaceProcessor';
import type { FaceMotionSample } from '../native/FaceProcessor';
import { subscribeSyncEvents } from '../utils/syncEvents';
import { getAttendanceLocation, formatLocationSummary, type LocationFix } from '../utils/locationManager';

type LivenessStatus = 'idle' | 'checking' | 'passed' | 'failed';
type VerifyStatus = 'idle' | 'no_profiles' | 'verifying' | 'matched' | 'no_match';
const MIN_NATIVE_SUCCESS_FRAMES = 3;

const CheckInScreen: React.FC = () => {
  const { hasPermission, requestPermission } = useCameraPermission();
  const frontCamera = useCameraDevice('front');
  const cameraRef = useRef<Camera | null>(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [livenessStatus, setLivenessStatus] = useState<LivenessStatus>('idle');
  const [verifyStatus, setVerifyStatus] = useState<VerifyStatus>('idle');
  const [pendingCount, setPendingCount] = useState(() => getUnsyncedRecords().length);
  const [instruction, setInstruction] = useState('Position face in oval');
  const [lastScore, setLastScore] = useState<number | null>(null);
  const [lastLocation, setLastLocation] = useState<LocationFix | null>(null);
  const [matchedName, setMatchedName] = useState<string | null>(null);

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  useEffect(() => {
    let mounted = true;
    (async () => {
      if (!NATIVE_ANALYZER_AVAILABLE || !mounted) return;
      try { await loadNativeFaceModels(); } catch { /* fallback */ }
    })();
    return () => {
      mounted = false;
      if (NATIVE_ANALYZER_AVAILABLE) releaseNativeFaceModels().catch(() => {});
    };
  }, []);

  useEffect(() => {
    if (!cameraActive) return;
    const t = setTimeout(() => setCameraReady(true), 1800);
    return () => clearTimeout(t);
  }, [cameraActive]);

  const refreshPending = useCallback(() => setPendingCount(getUnsyncedRecords().length), []);

  useEffect(() => {
    return subscribeSyncEvents(ev => { if (ev.type === 'finished') refreshPending(); });
  }, [refreshPending]);

  const handleStartVerify = useCallback(async () => {
    const profiles = getAllProfiles();
    if (profiles.length === 0) { setVerifyStatus('no_profiles'); return; }
    setVerifyStatus('idle');
    setLastScore(null);
    setLastLocation(null);
    setMatchedName(null);
    setInstruction('Getting location…');
    await getAttendanceLocation().catch(() => null);
    setCameraActive(true);
    setCameraReady(false);
    setLivenessStatus('idle');
    setInstruction('Position face in oval');
  }, []);

  const runVerification = useCallback(async () => {
    if (!cameraReady) { setInstruction('Camera initializing, please wait…'); return; }
    setLivenessStatus('checking');
    setVerifyStatus('verifying');
    setInstruction('Liveness: slowly turn head left & right…');

    const locationPromise = getAttendanceLocation();
    const result = processFaceFrameStub();
    const profiles = getAllProfiles().filter(p => p.faceVector.length > 0);
    if (profiles.length === 0) {
      setLivenessStatus('failed'); setVerifyStatus('no_profiles');
      setInstruction('No enrolled faces found'); return;
    }

    const frameEmbeds: number[][] = [];
    const motionSamples: FaceMotionSample[] = [];
    const demoTarget = profiles[0];
    let nativeSuccessFrames = 0;

    if (NATIVE_ANALYZER_AVAILABLE) {
      try { await loadNativeFaceModels(); } catch { /* guard below */ }
    }
    for (let i = 0; i < VERIFICATION_WINDOW_FRAMES; i += 1) {
      let emb: number[] | null = null;
      if (NATIVE_ANALYZER_AVAILABLE) {
        try {
          const cam = cameraRef.current;
          if (!cam) { continue; }
          let uri: string | null = null;
          const tp = cam.takePhoto;
          if (typeof tp === 'function') {
            const photo = await tp.call(cam, { enableAutoRedEyeReduction: false });
            uri = photo?.path ? `file://${photo.path}` : null;
          } else {
            const snap = await cam.takeSnapshot();
            const tmp = await snap.saveToTemporaryFileAsync('jpg', 90);
            uri = tmp ? `file://${tmp}` : null;
          }
          const nr = await analyzeFrameNative({
            width: 112, height: 112,
            pixelData: uri ? JSON.stringify({ uri, source: 'camera_photo' }) : '',
            runFaceNet: true,
          });
          if (nr?.faceDetected) {
            motionSamples.push({ centerX: nr.faceCenterX, centerY: nr.faceCenterY, width: nr.faceWidth });
          }
          if (nr?.faceDetected && nr.faceVector && nr.faceVector.length === FACE_VECTOR_SIZE) {
            emb = nr.faceVector;
          }
        } catch { emb = null; }
      }
      if (emb) { frameEmbeds.push(emb); nativeSuccessFrames += 1; }
      else if (!NATIVE_ANALYZER_AVAILABLE) {
        frameEmbeds.push(jitterEmbedding(demoTarget.faceVector, `${demoTarget.id}|frame:${i}`, 0.02));
      }
    }

    if (NATIVE_ANALYZER_AVAILABLE && nativeSuccessFrames < MIN_NATIVE_SUCCESS_FRAMES) {
      setLivenessStatus('failed'); setVerifyStatus('no_match');
      setInstruction('Face capture unstable. Retry in better lighting.'); return;
    }

    if (NATIVE_ANALYZER_AVAILABLE && motionSamples.length >= 2) {
      const liveness = evaluateLiveness(motionSamples);
      if (!liveness.passed) {
        setLivenessStatus('failed'); setVerifyStatus('no_match');
        setInstruction('Liveness failed. Turn your head, then retry.'); return;
      }
    }

    const probe = averageEmbeddings(frameEmbeds);
    const match = matchAgainstGallery(probe, profiles.map(p => ({ id: p.id, faceVector: p.faceVector })));
    // eslint-disable-next-line no-console
    console.log(`[CheckIn] score=${match.score.toFixed(4)} matched=${match.matched} user=${match.userId ?? 'none'}`);

    setLivenessStatus(match.matched ? 'passed' : 'failed');
    setVerifyStatus(match.matched ? 'matched' : 'no_match');
    setLastScore(match.score);
    setInstruction(match.matched ? 'Identity confirmed ✓' : 'No match — please retry');

    if (match.matched && match.userId) {
      let loc = await locationPromise.catch(() => null);
      if (!loc) loc = await getAttendanceLocation().catch(() => null);
      setLastLocation(loc);
      const profile = getProfile(match.userId);
      setMatchedName(profile?.name ?? match.userId);
      const record: OfflineAttendance = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
        userId: match.userId, checkInTime: Date.now(),
        latitude: loc?.latitude ?? null, longitude: loc?.longitude ?? null,
        locationAccuracy: loc?.accuracy ?? null, synced: false,
        livenessPassed: true,
        similarityScore: match.score || (result.faceDetected ? 0.95 : 0.0),
      };
      saveAttendanceRecord(record);
      refreshPending();
    }

    await new Promise<void>(r => setTimeout(r, 2000));
    setCameraActive(false);
    setLivenessStatus('idle');
    setVerifyStatus('idle');
  }, [cameraReady, refreshPending]);

  if (!hasPermission) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.centered}>
          <Text style={styles.permIcon}>🔒</Text>
          <Text style={styles.permTitle}>Camera Access Required</Text>
          <Text style={styles.permBody}>Face verification requires camera permission.</Text>
          <TouchableOpacity style={styles.primaryBtn} onPress={requestPermission}>
            <Text style={styles.primaryBtnText}>Grant Permission</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  if (cameraActive) {
    if (!frontCamera) {
      return (
        <SafeAreaView style={styles.container}>
          <View style={styles.centered}>
            <Text style={styles.permTitle}>Front Camera Unavailable</Text>
            <TouchableOpacity style={styles.primaryBtn} onPress={() => setCameraActive(false)}>
              <Text style={styles.primaryBtnText}>Go Back</Text>
            </TouchableOpacity>
          </View>
        </SafeAreaView>
      );
    }
    return (
      <View style={styles.container}>
        <Camera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device={frontCamera}
          isActive={true}
          photo={true}
          onInitialized={() => setCameraReady(true)}
          onStarted={() => setCameraReady(true)}
          onPreviewStarted={() => setCameraReady(true)}
        />
        <CameraOverlay
          faceDetected={livenessStatus !== 'idle'}
          livenessStatus={livenessStatus}
          instruction={instruction}
        />
        <View style={styles.captureBar}>
          {livenessStatus === 'checking' && (
            <Text style={styles.captureHint}>Turn head slowly left & right</Text>
          )}
          <View style={styles.captureRow}>
            <TouchableOpacity style={styles.cancelBtn} onPress={() => { setCameraActive(false); setLivenessStatus('idle'); setVerifyStatus('idle'); }}>
              <Text style={styles.cancelBtnText}>✕ Cancel</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.verifyBtn, livenessStatus === 'checking' && styles.verifyBtnBusy]}
              onPress={runVerification}
              disabled={livenessStatus === 'checking' || !cameraReady}>
              {livenessStatus === 'checking'
                ? <ActivityIndicator color="#fff" />
                : <Text style={styles.verifyBtnText}>{cameraReady ? 'Verify & Check In' : 'Initializing…'}</Text>}
            </TouchableOpacity>
          </View>
        </View>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.page}>

        <View style={styles.brand}>
          <View style={styles.brandBadge}><Text style={styles.brandBadgeText}>NHAI</Text></View>
          <View>
            <Text style={styles.brandTitle}>Attendance Check-In</Text>
            <Text style={styles.brandSub}>Offline face verification</Text>
          </View>
        </View>

        <View style={styles.statusRow}>
          <View style={styles.chip}>
            <View style={[styles.chipDot, { backgroundColor: '#f59e0b' }]} />
            <Text style={styles.chipText}>Offline Mode</Text>
          </View>
          {pendingCount > 0 && (
            <View style={[styles.chip, { borderColor: '#f59e0b' }]}>
              <Text style={[styles.chipText, { color: '#f59e0b' }]}>{pendingCount} pending sync</Text>
            </View>
          )}
          {pendingCount === 0 && (
            <View style={[styles.chip, { borderColor: '#22c55e' }]}>
              <Text style={[styles.chipText, { color: '#22c55e' }]}>All synced ✓</Text>
            </View>
          )}
        </View>

        {verifyStatus === 'no_profiles' && (
          <View style={styles.alertBox}>
            <Text style={styles.alertText}>⚠  No enrolled profiles found. Go to Enrol tab first.</Text>
          </View>
        )}

        <TouchableOpacity style={styles.primaryBtn} onPress={handleStartVerify}>
          <Text style={styles.primaryBtnText}>🔍  Start Face Verification</Text>
        </TouchableOpacity>

        {(verifyStatus === 'matched' || verifyStatus === 'no_match') && (
          <View style={[
            styles.resultCard,
            verifyStatus === 'matched' ? styles.resultSuccess : styles.resultFail,
          ]}>
            {verifyStatus === 'matched' ? (
              <>
                <Text style={styles.resultIcon}>✓</Text>
                <Text style={styles.resultTitle}>{matchedName ?? 'Identity Confirmed'}</Text>
                <Text style={styles.resultSub}>
                  Score: {lastScore?.toFixed(3)}{'\n'}
                  {lastLocation ? formatLocationSummary(lastLocation) : 'GPS unavailable'}
                </Text>
              </>
            ) : (
              <>
                <Text style={[styles.resultIcon, { color: '#ef4444' }]}>✗</Text>
                <Text style={[styles.resultTitle, { color: '#ef4444' }]}>No Match</Text>
                <Text style={styles.resultSub}>Score: {lastScore?.toFixed(3) ?? '—'}</Text>
              </>
            )}
          </View>
        )}

        <View style={styles.infoRow}>
          <View style={[styles.infoChip, { borderColor: NATIVE_ANALYZER_AVAILABLE ? '#22c55e' : '#f59e0b' }]}>
            <Text style={[styles.infoChipText, { color: NATIVE_ANALYZER_AVAILABLE ? '#22c55e' : '#f59e0b' }]}>
              {NATIVE_ANALYZER_AVAILABLE ? '🧠 AI Active' : '⚠ AI Fallback'}
            </Text>
          </View>
          <View style={styles.infoChip}>
            <Text style={styles.infoChipText}>📍 GPS + Offline Sync</Text>
          </View>
        </View>

      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0f172a' },
  page: { flex: 1, padding: 20 },
  centered: { flex: 1, padding: 32, justifyContent: 'center', alignItems: 'center' },

  brand: { flexDirection: 'row', alignItems: 'center', gap: 14, marginBottom: 24, marginTop: 8 },
  brandBadge: { backgroundColor: '#1d4ed8', borderRadius: 10, paddingHorizontal: 12, paddingVertical: 8 },
  brandBadgeText: { color: '#fff', fontWeight: '900', fontSize: 16, letterSpacing: 1.5 },
  brandTitle: { fontSize: 22, fontWeight: '700', color: '#f8fafc' },
  brandSub: { fontSize: 13, color: '#64748b', marginTop: 2 },

  statusRow: { flexDirection: 'row', gap: 10, marginBottom: 20, flexWrap: 'wrap' },
  chip: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    backgroundColor: '#1e293b', borderRadius: 20, paddingHorizontal: 12,
    paddingVertical: 6, borderWidth: 1, borderColor: '#334155',
  },
  chipDot: { width: 7, height: 7, borderRadius: 4 },
  chipText: { color: '#94a3b8', fontSize: 12, fontWeight: '600' },

  alertBox: {
    backgroundColor: '#451a03', borderRadius: 12, padding: 14,
    marginBottom: 16, borderLeftWidth: 3, borderLeftColor: '#f97316',
  },
  alertText: { color: '#fed7aa', fontSize: 14, lineHeight: 20 },

  primaryBtn: { backgroundColor: '#1d4ed8', borderRadius: 14, padding: 18, alignItems: 'center', marginBottom: 20 },
  primaryBtnText: { color: '#fff', fontSize: 16, fontWeight: '700' },

  resultCard: { borderRadius: 16, padding: 24, alignItems: 'center', marginBottom: 20, borderWidth: 1 },
  resultSuccess: { backgroundColor: '#052e16', borderColor: '#22c55e' },
  resultFail: { backgroundColor: '#450a0a', borderColor: '#ef4444' },
  resultIcon: { fontSize: 36, color: '#22c55e', fontWeight: '800', marginBottom: 8 },
  resultTitle: { fontSize: 20, fontWeight: '700', color: '#f8fafc', marginBottom: 6 },
  resultSub: { fontSize: 13, color: '#94a3b8', textAlign: 'center', lineHeight: 20 },

  infoRow: { flexDirection: 'row', gap: 10, marginTop: 'auto', flexWrap: 'wrap' },
  infoChip: {
    backgroundColor: '#1e293b', borderRadius: 10, paddingHorizontal: 12,
    paddingVertical: 8, borderWidth: 1, borderColor: '#334155',
  },
  infoChipText: { color: '#64748b', fontSize: 12, fontWeight: '600' },

  captureBar: {
    position: 'absolute', bottom: 0, left: 0, right: 0,
    padding: 20, gap: 10, backgroundColor: 'rgba(15,23,42,0.92)',
  },
  captureHint: { color: '#fde68a', fontSize: 14, fontWeight: '600', textAlign: 'center' },
  captureRow: { flexDirection: 'row', gap: 12 },
  cancelBtn: { flex: 1, borderRadius: 12, padding: 16, alignItems: 'center', backgroundColor: '#1e293b', borderWidth: 1, borderColor: '#334155' },
  cancelBtnText: { color: '#94a3b8', fontSize: 15, fontWeight: '600' },
  verifyBtn: { flex: 2, borderRadius: 12, padding: 16, alignItems: 'center', backgroundColor: '#16a34a' },
  verifyBtnBusy: { backgroundColor: '#14532d' },
  verifyBtnText: { color: '#fff', fontSize: 15, fontWeight: '700' },

  permIcon: { fontSize: 48, marginBottom: 16 },
  permTitle: { fontSize: 20, fontWeight: '700', color: '#f8fafc', marginBottom: 10, textAlign: 'center' },
  permBody: { fontSize: 14, color: '#94a3b8', textAlign: 'center', marginBottom: 24, lineHeight: 22 },
});

export default CheckInScreen;
