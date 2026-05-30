import NetInfo from '@react-native-community/netinfo';
import {
  getUnsyncedRecords,
  getAllAttendanceRecords,
  markRecordsSynced,
  purgeSyncedRecords,
  getProfile,
  type OfflineAttendance,
} from '../db/mmkv';
import {
  isSyncEndpointConfigured,
  SYNC_CONFIG,
  USE_LOCAL_SYNC_SERVER,
} from '../config/syncConfig';
import type { NetInfoState } from '@react-native-community/netinfo';

export type SyncStatus = 'idle' | 'syncing' | 'success' | 'offline' | 'error' | 'demo';

export interface SyncResult {
  status: SyncStatus;
  syncedCount: number;
  message: string;
}

export interface AttendanceUploadPayload {
  records: Array<{
    id: string;
    userId: string;
    employeeCode: string | null;
    name: string | null;
    checkInTime: number;
    latitude: number | null;
    longitude: number | null;
    locationAccuracy: number | null;
    livenessPassed: boolean;
    similarityScore: number;
  }>;
  syncedAt: string;
  source: 'datalake-attendance-mobile';
}

let syncInProgress = false;

export function buildUploadPayload(
  records: OfflineAttendance[],
): AttendanceUploadPayload {
  return {
    records: records.map(r => {
      const profile = getProfile(r.userId);
      return {
        id: r.id,
        userId: r.userId,
        employeeCode: profile?.employeeCode ?? null,
        name: profile?.name ?? null,
        checkInTime: r.checkInTime,
        latitude: r.latitude,
        longitude: r.longitude,
        locationAccuracy: r.locationAccuracy,
        livenessPassed: r.livenessPassed,
        similarityScore: r.similarityScore,
      };
    }),
    syncedAt: new Date().toISOString(),
    source: 'datalake-attendance-mobile',
  };
}

/** LAN local server works on WiFi without public internet. */
export function isNetAvailableForSync(state: NetInfoState): boolean {
  if (state.isConnected !== true) {
    return false;
  }
  if (USE_LOCAL_SYNC_SERVER) {
    return true;
  }
  return state.isInternetReachable !== false;
}

export async function isDeviceOnline(): Promise<boolean> {
  try {
    const state = await NetInfo.fetch();
    return isNetAvailableForSync(state);
  } catch {
    return false;
  }
}

async function postToAws(payload: AttendanceUploadPayload): Promise<void> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), SYNC_CONFIG.timeoutMs);

  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    };
    if (SYNC_CONFIG.apiKey.trim()) {
      headers.Authorization = `Bearer ${SYNC_CONFIG.apiKey.trim()}`;
    }

    const response = await fetch(SYNC_CONFIG.apiUrl.trim(), {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!response.ok) {
      const body = await response.text().catch(() => '');
      throw new Error(`HTTP ${response.status}${body ? `: ${body.slice(0, 120)}` : ''}`);
    }
  } finally {
    clearTimeout(timeout);
  }
}

async function demoSync(records: OfflineAttendance[]): Promise<SyncResult> {
  await new Promise<void>(r => setTimeout(r, 400));
  const ids = records.map(r => r.id);
  markRecordsSynced(ids);
  purgeSyncedRecords();
  return {
    status: 'demo',
    syncedCount: ids.length,
    message: `Demo sync: ${ids.length} record(s) uploaded (configure apiUrl for real AWS POST).`,
  };
}

/**
 * Upload unsynced attendance to AWS (or demo mode), mark synced, purge on success.
 */
export async function syncUnsyncedAttendance(
  options: { force?: boolean } = {},
): Promise<SyncResult> {
  if (syncInProgress) {
    return { status: 'idle', syncedCount: 0, message: 'Sync already in progress.' };
  }

  const pending = getUnsyncedRecords();
  if (pending.length === 0) {
    return { status: 'idle', syncedCount: 0, message: 'Nothing to sync.' };
  }

  syncInProgress = true;
  try {
    const online = options.force ? true : await isDeviceOnline();
    if (!online && isSyncEndpointConfigured()) {
      return {
        status: 'offline',
        syncedCount: 0,
        message: 'No internet connection. Sync will retry when online.',
      };
    }

    const payload = buildUploadPayload(pending);
    const ids = pending.map(r => r.id);

    if (!isSyncEndpointConfigured()) {
      if (!SYNC_CONFIG.demoModeWhenNoUrl) {
        return {
          status: 'error',
          syncedCount: 0,
          message: 'Set SYNC_CONFIG.apiUrl in src/config/syncConfig.ts',
        };
      }
      return demoSync(pending);
    }

    if (!online) {
      return {
        status: 'offline',
        syncedCount: 0,
        message: 'No internet connection.',
      };
    }

    await postToAws(payload);
    markRecordsSynced(ids);
    purgeSyncedRecords();

    return {
      status: 'success',
      syncedCount: ids.length,
      message: `Synced ${ids.length} record(s) to AWS.`,
    };
  } catch (e) {
    const msg = e instanceof Error ? e.message : 'Sync failed';
    return { status: 'error', syncedCount: 0, message: msg };
  } finally {
    syncInProgress = false;
  }
}

export function countSyncedPendingPurge(): number {
  return getAllAttendanceRecords().filter(r => r.synced).length;
}

/** Remove records already marked synced; returns how many were removed. */
export function purgeSyncedOnly(): number {
  const count = countSyncedPendingPurge();
  if (count > 0) {
    purgeSyncedRecords();
  }
  return count;
}
