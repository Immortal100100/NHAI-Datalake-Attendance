import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  TouchableOpacity,
  TextInput,
  Alert,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import CameraOverlay from '../components/CameraOverlay';
import { saveProfile, getAllProfiles, type UserProfile } from '../db/mmkv';
import {
  FACE_VECTOR_SIZE,
  NATIVE_ANALYZER_AVAILABLE,
  analyzeFrameNative,
  loadNativeFaceModels,
  releaseNativeFaceModels,
  syntheticFaceVector,
} from '../native/FaceProcessor';

const RegisterScreen: React.FC = () => {
  const { hasPermission, requestPermission } = useCameraPermission();
  const frontCamera = useCameraDevice('front');
  const cameraRef = useRef<Camera | null>(null);

  const [userId, setUserId] = useState('');
  const [userName, setUserName] = useState('');
  const [employeeCode, setEmployeeCode] = useState('');
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [enrolledCount, setEnrolledCount] = useState(() => getAllProfiles().length);

  useEffect(() => {
    if (!cameraActive) return;
    const t = setTimeout(() => setCameraReady(true), 1800);
    return () => clearTimeout(t);
  }, [cameraActive]);

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

  const handleStartCapture = useCallback(() => {
    if (!userId.trim() || !userName.trim() || !employeeCode.trim()) {
      Alert.alert('Missing Fields', 'Please fill in all three fields before capturing.');
      return;
    }
    setCameraReady(false);
    setCameraActive(true);
  }, [userId, userName, employeeCode]);

  const handleCaptureBiometrics = useCallback(async () => {
    if (!cameraReady) {
      Alert.alert('Camera Not Ready', 'Please wait a moment and try again.');
      return;
    }
    setCapturing(true);
    try {
      await new Promise<void>(r => setTimeout(r, 400));
      let faceVector = syntheticFaceVector(`${userId.trim()}|${employeeCode.trim()}`);
      let vectorSource: 'synthetic' | 'native' = 'synthetic';
      let nativeFailureReason = 'Unknown error';

      if (NATIVE_ANALYZER_AVAILABLE) {
        await loadNativeFaceModels();
        for (let attempt = 1; attempt <= 3; attempt += 1) {
          try {
            const camera = cameraRef.current;
            if (!camera) { await new Promise<void>(r => setTimeout(r, 220)); continue; }
            let uri: string | null = null;
            const takePhoto = camera.takePhoto;
            if (typeof takePhoto === 'function') {
              const photo = await takePhoto.call(camera, { enableAutoRedEyeReduction: false });
              uri = photo?.path ? `file://${photo.path}` : null;
            } else {
              const snap = await camera.takeSnapshot();
              const tmp = await snap.saveToTemporaryFileAsync('jpg', 90);
              uri = tmp ? `file://${tmp}` : null;
            }
            if (!uri) { nativeFailureReason = `Empty path (attempt ${attempt})`; await new Promise<void>(r => setTimeout(r, 220)); continue; }
            const res = await analyzeFrameNative({ width: 112, height: 112, pixelData: JSON.stringify({ uri, source: 'camera_photo' }), runFaceNet: true });
            if (res?.faceDetected && res.faceVector && res.faceVector.length === FACE_VECTOR_SIZE) {
              faceVector = res.faceVector;
              vectorSource = 'native';
              break;
            }
            nativeFailureReason = `Face not detected (attempt ${attempt})`;
          } catch (e) { nativeFailureReason = `Error: ${String(e)}`; }
          await new Promise<void>(r => setTimeout(r, 220));
        }
      }

      if (NATIVE_ANALYZER_AVAILABLE && vectorSource !== 'native') {
        Alert.alert('Enrollment Failed', `${nativeFailureReason}.\n\nPlease ensure good lighting and face centered in oval.`);
        setCapturing(false);
        return;
      }

      const profile: UserProfile = {
        id: userId.trim(),
        name: userName.trim(),
        employeeCode: employeeCode.trim(),
        faceVector,
        createdAt: Date.now(),
      };
      saveProfile(profile);
      setEnrolledCount(getAllProfiles().length);

      Alert.alert('Enrolled', `${userName.trim()} registered successfully.`, [
        { text: 'Done', onPress: () => { setCameraActive(false); setUserId(''); setUserName(''); setEmployeeCode(''); } },
      ]);
    } catch {
      Alert.alert('Error', 'Failed to capture biometrics. Please try again.');
    } finally {
      setCapturing(false);
    }
  }, [cameraReady, userId, userName, employeeCode]);

  if (!hasPermission) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.centered}>
          <Text style={styles.permIcon}>🔒</Text>
          <Text style={styles.permTitle}>Camera Access Required</Text>
          <Text style={styles.permBody}>Biometric enrollment requires camera access.</Text>
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
            <Text style={styles.permIcon}>📷</Text>
            <Text style={styles.permTitle}>Front Camera Unavailable</Text>
            <Text style={styles.permBody}>Close other camera apps and retry.</Text>
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
          faceDetected={false}
          livenessStatus="idle"
          instruction="Centre face in oval • Remove glasses • Good lighting"
        />
        <View style={styles.captureBar}>
          <TouchableOpacity style={styles.cancelBtn} onPress={() => setCameraActive(false)}>
            <Text style={styles.cancelBtnText}>✕ Cancel</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.captureBtn, capturing && styles.captureBtnBusy]}
            onPress={handleCaptureBiometrics}
            disabled={capturing || !cameraReady}>
            {capturing
              ? <ActivityIndicator color="#fff" />
              : <Text style={styles.captureBtnText}>{cameraReady ? 'Capture Face' : 'Initializing…'}</Text>}
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">

        <View style={styles.brand}>
          <View style={styles.brandBadge}>
            <Text style={styles.brandBadgeText}>NHAI</Text>
          </View>
          <View>
            <Text style={styles.brandTitle}>Biometric Enrolment</Text>
            <Text style={styles.brandSub}>Register field personnel</Text>
          </View>
        </View>

        <View style={styles.statsRow}>
          <View style={styles.statPill}>
            <Text style={styles.statNum}>{enrolledCount}</Text>
            <Text style={styles.statLbl}>Enrolled</Text>
          </View>
          <View style={[styles.statPill, { borderColor: NATIVE_ANALYZER_AVAILABLE ? '#22c55e' : '#f59e0b' }]}>
            <Text style={[styles.statNum, { color: NATIVE_ANALYZER_AVAILABLE ? '#22c55e' : '#f59e0b' }]}>
              {NATIVE_ANALYZER_AVAILABLE ? 'ON' : 'OFF'}
            </Text>
            <Text style={styles.statLbl}>AI Model</Text>
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.fieldLabel}>USER ID</Text>
          <TextInput
            style={styles.input}
            value={userId}
            onChangeText={setUserId}
            placeholder="e.g. EMP001"
            placeholderTextColor="#475569"
            autoCapitalize="characters"
          />

          <Text style={styles.fieldLabel}>FULL NAME</Text>
          <TextInput
            style={styles.input}
            value={userName}
            onChangeText={setUserName}
            placeholder="e.g. Rajesh Kumar"
            placeholderTextColor="#475569"
          />

          <Text style={styles.fieldLabel}>EMPLOYEE CODE</Text>
          <TextInput
            style={[styles.input, { marginBottom: 0 }]}
            value={employeeCode}
            onChangeText={setEmployeeCode}
            placeholder="e.g. NHAI-2024-001"
            placeholderTextColor="#475569"
            autoCapitalize="characters"
          />
        </View>

        <TouchableOpacity
          style={[styles.primaryBtn, styles.primaryBtnLarge]}
          onPress={handleStartCapture}>
          <Text style={styles.primaryBtnText}>📷  Open Camera & Enrol</Text>
        </TouchableOpacity>

        <View style={styles.tipBox}>
          <Text style={styles.tipText}>
            💡  Ensure good lighting, remove glasses, and keep your face centred in the oval for best results.
          </Text>
        </View>

      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0f172a' },
  scroll: { flexGrow: 1, padding: 20 },
  centered: { flex: 1, padding: 32, justifyContent: 'center', alignItems: 'center' },

  // Brand header
  brand: { flexDirection: 'row', alignItems: 'center', gap: 14, marginBottom: 24, marginTop: 8 },
  brandBadge: { backgroundColor: '#1d4ed8', borderRadius: 10, paddingHorizontal: 12, paddingVertical: 8 },
  brandBadgeText: { color: '#fff', fontWeight: '900', fontSize: 16, letterSpacing: 1.5 },
  brandTitle: { fontSize: 22, fontWeight: '700', color: '#f8fafc' },
  brandSub: { fontSize: 13, color: '#64748b', marginTop: 2 },

  // Stats
  statsRow: { flexDirection: 'row', gap: 12, marginBottom: 20 },
  statPill: {
    flex: 1, backgroundColor: '#1e293b', borderRadius: 12,
    padding: 14, alignItems: 'center', borderWidth: 1, borderColor: '#334155',
  },
  statNum: { fontSize: 22, fontWeight: '800', color: '#3b82f6' },
  statLbl: { fontSize: 11, color: '#64748b', marginTop: 2, textTransform: 'uppercase', letterSpacing: 0.5 },

  // Form card
  card: { backgroundColor: '#1e293b', borderRadius: 16, padding: 20, marginBottom: 20, borderWidth: 1, borderColor: '#334155' },
  fieldLabel: { fontSize: 11, fontWeight: '700', color: '#475569', marginBottom: 6, letterSpacing: 1, textTransform: 'uppercase' },
  input: {
    backgroundColor: '#0f172a', borderRadius: 10, padding: 14,
    fontSize: 15, color: '#f8fafc', marginBottom: 18,
    borderWidth: 1, borderColor: '#334155',
  },

  // Buttons
  primaryBtn: { backgroundColor: '#1d4ed8', borderRadius: 14, padding: 16, alignItems: 'center', marginBottom: 16 },
  primaryBtnLarge: { padding: 18 },
  primaryBtnText: { color: '#fff', fontSize: 16, fontWeight: '700' },

  // Camera bar
  captureBar: {
    position: 'absolute', bottom: 0, left: 0, right: 0,
    flexDirection: 'row', padding: 20, gap: 12,
    backgroundColor: 'rgba(15,23,42,0.92)',
  },
  cancelBtn: { flex: 1, borderRadius: 12, padding: 16, alignItems: 'center', backgroundColor: '#1e293b', borderWidth: 1, borderColor: '#334155' },
  cancelBtnText: { color: '#94a3b8', fontSize: 15, fontWeight: '600' },
  captureBtn: { flex: 2, borderRadius: 12, padding: 16, alignItems: 'center', backgroundColor: '#1d4ed8' },
  captureBtnBusy: { backgroundColor: '#1e3a8a' },
  captureBtnText: { color: '#fff', fontSize: 15, fontWeight: '700' },

  // Tip
  tipBox: { backgroundColor: '#1e293b', borderRadius: 12, padding: 14, borderLeftWidth: 3, borderLeftColor: '#3b82f6' },
  tipText: { color: '#94a3b8', fontSize: 13, lineHeight: 20 },

  // Permission
  permIcon: { fontSize: 48, marginBottom: 16 },
  permTitle: { fontSize: 20, fontWeight: '700', color: '#f8fafc', marginBottom: 10, textAlign: 'center' },
  permBody: { fontSize: 14, color: '#94a3b8', textAlign: 'center', marginBottom: 24, lineHeight: 22 },
});

export default RegisterScreen;
