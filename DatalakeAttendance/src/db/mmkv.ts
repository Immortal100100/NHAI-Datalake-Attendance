import { createMMKV, type MMKV } from 'react-native-mmkv';

// AES-256 encrypted storage instance
export const storage: MMKV = createMMKV({
  id: 'datalake-secure-store',
  encryptionKey: 'DL3_NHAI_SECURE_KEY_2026',
  encryptionType: 'AES-256',
});

// ─── TypeScript Interfaces ────────────────────────────────────────────────────

export interface UserProfile {
  id: string;             // Unique employee / user ID
  name: string;           // Full name
  employeeCode: string;   // NHAI employee code
  faceVector: number[];   // 128-D MobileFaceNet embedding
  createdAt: number;      // Unix timestamp (ms)
}

export interface OfflineAttendance {
  id: string;             // UUID for ledger entry
  userId: string;         // References UserProfile.id
  checkInTime: number;    // Unix timestamp (ms)
  latitude: number | null;
  longitude: number | null;
  locationAccuracy: number | null;
  synced: boolean;        // false until successfully POSTed to AWS
  livenessPassed: boolean;
  similarityScore: number; // Cosine similarity value at check-in
}

// ─── Storage Keys ─────────────────────────────────────────────────────────────

const PROFILES_KEY = 'user_profiles';
const ATTENDANCE_KEY = 'offline_attendance';

// ─── UserProfile CRUD ─────────────────────────────────────────────────────────

export function saveProfile(profile: UserProfile): void {
  const profiles = getAllProfiles();
  const idx = profiles.findIndex(p => p.id === profile.id);
  if (idx >= 0) {
    profiles[idx] = profile;
  } else {
    profiles.push(profile);
  }
  storage.set(PROFILES_KEY, JSON.stringify(profiles));
}

export function getProfile(id: string): UserProfile | null {
  const profiles = getAllProfiles();
  return profiles.find(p => p.id === id) ?? null;
}

export function getAllProfiles(): UserProfile[] {
  const raw = storage.getString(PROFILES_KEY);
  if (!raw) return [];
  try {
    return JSON.parse(raw) as UserProfile[];
  } catch {
    return [];
  }
}

export function deleteProfile(id: string): void {
  const profiles = getAllProfiles().filter(p => p.id !== id);
  storage.set(PROFILES_KEY, JSON.stringify(profiles));
}

// ─── OfflineAttendance Ledger ─────────────────────────────────────────────────

export function saveAttendanceRecord(record: OfflineAttendance): void {
  const records = getAllAttendanceRecords();
  const idx = records.findIndex(r => r.id === record.id);
  if (idx >= 0) {
    records[idx] = record;
  } else {
    records.push(record);
  }
  storage.set(ATTENDANCE_KEY, JSON.stringify(records));
}

export function getAllAttendanceRecords(): OfflineAttendance[] {
  const raw = storage.getString(ATTENDANCE_KEY);
  if (!raw) return [];
  try {
    return JSON.parse(raw) as OfflineAttendance[];
  } catch {
    return [];
  }
}

export function getUnsyncedRecords(): OfflineAttendance[] {
  return getAllAttendanceRecords().filter(r => !r.synced);
}

export function markRecordsSynced(ids: string[]): void {
  const records = getAllAttendanceRecords().map(r =>
    ids.includes(r.id) ? { ...r, synced: true } : r,
  );
  storage.set(ATTENDANCE_KEY, JSON.stringify(records));
}

export function purgeSyncedRecords(): void {
  const unsynced = getAllAttendanceRecords().filter(r => !r.synced);
  storage.set(ATTENDANCE_KEY, JSON.stringify(unsynced));
}

export function clearAllData(): void {
  storage.remove(PROFILES_KEY);
  storage.remove(ATTENDANCE_KEY);
}
