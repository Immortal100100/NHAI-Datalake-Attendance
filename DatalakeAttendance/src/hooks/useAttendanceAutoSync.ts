import { useEffect, useRef } from 'react';
import { AppState, type AppStateStatus } from 'react-native';
import NetInfo, { type NetInfoState } from '@react-native-community/netinfo';
import { syncUnsyncedAttendance, isNetAvailableForSync } from '../utils/attendanceSync';
import { getUnsyncedRecords } from '../db/mmkv';
import { isSyncEndpointConfigured, SYNC_CONFIG } from '../config/syncConfig';
import { emitSyncEvent } from '../utils/syncEvents';

/**
 * Auto-uploads unsynced attendance when WiFi/network connects or app opens online.
 * No manual tap required.
 */
export function useAttendanceAutoSync(
  onSyncComplete?: (syncedCount: number) => void,
): void {
  const lastRunRef = useRef(0);
  const wasConnectedRef = useRef<boolean | null>(null);

  useEffect(() => {
    const runIfNeeded = async (reason: string) => {
      if (getUnsyncedRecords().length === 0) {
        return;
      }
      const now = Date.now();
      if (now - lastRunRef.current < 3000) {
        return;
      }
      lastRunRef.current = now;

      emitSyncEvent({ type: 'started' });
      try {
        const result = await syncUnsyncedAttendance();
        emitSyncEvent({ type: 'finished', result });
        if (result.syncedCount > 0) {
          onSyncComplete?.(result.syncedCount);
        }
        // eslint-disable-next-line no-console
        console.log(`[AutoSync] ${reason} → ${result.status}: ${result.message}`);
      } catch (e) {
        const msg = e instanceof Error ? e.message : 'Sync failed';
        const result = {
          status: 'error' as const,
          syncedCount: 0,
          message: msg,
        };
        emitSyncEvent({ type: 'finished', result });
      }
    };

    const onConnectivity = (state: NetInfoState) => {
      const available = isNetAvailableForSync(state);
      const prev = wasConnectedRef.current;
      wasConnectedRef.current = available;

      if (available && prev !== true) {
        runIfNeeded('network_connected').catch(() => {});
      }
    };

    const onAppState = (next: AppStateStatus) => {
      if (next === 'active') {
        NetInfo.fetch()
          .then(state => {
            if (isNetAvailableForSync(state)) {
              return runIfNeeded('app_foreground');
            }
            return undefined;
          })
          .catch(() => {});
      }
    };

    const unsubNet = NetInfo.addEventListener(onConnectivity);
    const unsubApp = AppState.addEventListener('change', onAppState);

    NetInfo.fetch()
      .then(state => {
        wasConnectedRef.current = isNetAvailableForSync(state);
        if (wasConnectedRef.current) {
          return runIfNeeded('app_launch');
        }
        return undefined;
      })
      .catch(() => {});

    return () => {
      unsubNet();
      unsubApp.remove();
    };
  }, [onSyncComplete]);
}

export function getAutoSyncHint(): string {
  if (!isSyncEndpointConfigured()) {
    return 'Configure apiUrl in syncConfig.ts';
  }
  const url = SYNC_CONFIG.apiUrl;
  if (
    url.includes('192.168.') ||
    url.includes('10.0.2.2') ||
    url.includes('localhost')
  ) {
    return 'Syncs automatically when you open the app on WiFi (same network as PC server)';
  }
  return 'Syncs automatically when internet is available — no button tap needed';
}
