import React, { useState, useCallback, useEffect } from 'react';
import {
  View, Text, StyleSheet, SafeAreaView, TouchableOpacity,
  ScrollView, Alert, ActivityIndicator,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import NetInfo from '@react-native-community/netinfo';
import { getAllProfiles, getUnsyncedRecords, getAllAttendanceRecords, clearAllData } from '../db/mmkv';
import { purgeSyncedOnly, countSyncedPendingPurge, isDeviceOnline, type SyncResult } from '../utils/attendanceSync';
import { isSyncEndpointConfigured, SYNC_CONFIG, USE_LOCAL_SYNC_SERVER } from '../config/syncConfig';
import { getAutoSyncHint } from '../hooks/useAttendanceAutoSync';
import { subscribeSyncEvents } from '../utils/syncEvents';

const AdminSyncScreen: React.FC = () => {
  const [stats, setStats] = useState(() => ({
    profiles: getAllProfiles().length,
    totalRecords: getAllAttendanceRecords().length,
    unsynced: getUnsyncedRecords().length,
    syncedLocal: countSyncedPendingPurge(),
  }));
  const [online, setOnline] = useState<boolean | null>(null);
  const [autoSyncing, setAutoSyncing] = useState(false);
  const [lastResult, setLastResult] = useState<SyncResult | null>(null);

  const refresh = useCallback(() => {
    setStats({
      profiles: getAllProfiles().length,
      totalRecords: getAllAttendanceRecords().length,
      unsynced: getUnsyncedRecords().length,
      syncedLocal: countSyncedPendingPurge(),
    });
  }, []);

  useFocusEffect(useCallback(() => { refresh(); }, [refresh]));

  useEffect(() => {
    const updateOnline = async () => setOnline(await isDeviceOnline());
    updateOnline().catch(() => setOnline(false));
    const unsubNet = NetInfo.addEventListener(() => updateOnline().catch(() => setOnline(false)));
    const unsubSync = subscribeSyncEvents(ev => {
      if (ev.type === 'started') { setAutoSyncing(true); }
      else { setAutoSyncing(false); setLastResult(ev.result); refresh(); }
    });
    return () => { unsubNet(); unsubSync(); };
  }, [refresh]);

  const handlePurgeSynced = useCallback(() => {
    const n = countSyncedPendingPurge();
    if (n === 0) { Alert.alert('Nothing to purge', 'No synced records on device.'); return; }
    Alert.alert('Purge synced records', `Remove ${n} already-uploaded record(s) from this device?`, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Purge', onPress: () => { purgeSyncedOnly(); refresh(); } },
    ]);
  }, [refresh]);

  const handlePurgeAll = useCallback(() => {
    Alert.alert('Purge All Data', 'This permanently deletes all profiles and attendance records from this device.', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Purge', style: 'destructive', onPress: () => { clearAllData(); refresh(); setLastResult(null); } },
    ]);
  }, [refresh]);

  const endpointLabel = !isSyncEndpointConfigured()
    ? 'Demo mode (configure apiUrl in syncConfig.ts)'
    : USE_LOCAL_SYNC_SERVER
      ? `Local server: ${SYNC_CONFIG.apiUrl}`
      : `AWS: ${SYNC_CONFIG.apiUrl}`;

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scroll}>

        <View style={styles.brand}>
          <View style={styles.brandBadge}><Text style={styles.brandBadgeText}>NHAI</Text></View>
          <View>
            <Text style={styles.brandTitle}>Data Sync Panel</Text>
            <Text style={styles.brandSub}>Attendance records → AWS</Text>
          </View>
        </View>

        {/* Network status */}
        <View style={[styles.netBanner, online ? styles.netOnline : styles.netOffline]}>
          <View style={[styles.netDot,
            online === true ? styles.dotGreen : online === false ? styles.dotRed : styles.dotGrey]} />
          <Text style={styles.netText}>
            {online === null
              ? 'Checking network…'
              : online
                ? autoSyncing ? 'Online — syncing…' : 'Online — auto-sync ready'
                : 'Offline — records queued on device'}
          </Text>
          {autoSyncing && <ActivityIndicator color="#3b82f6" size="small" />}
        </View>

        {/* Stats */}
        <View style={styles.statsRow}>
          <StatCard label="Profiles" value={stats.profiles} color="#3b82f6" icon="👤" />
          <StatCard label="Records" value={stats.totalRecords} color="#8b5cf6" icon="📋" />
          <StatCard label="Pending" value={stats.unsynced} color="#f59e0b" icon="⏳" />
        </View>

        {/* Sync info */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Text style={styles.cardTitle}>Automatic Sync</Text>
            <View style={[styles.badge, { backgroundColor: '#052e16' }]}>
              <Text style={[styles.badgeText, { color: '#22c55e' }]}>Active</Text>
            </View>
          </View>
          <Text style={styles.cardBody}>{getAutoSyncHint()}</Text>
          <Text style={styles.cardHint}>{endpointLabel}</Text>
        </View>

        {/* Last result */}
        {lastResult && lastResult.status !== 'idle' && (
          <View style={[styles.resultBox,
            lastResult.status === 'error' && styles.resultError,
            lastResult.status === 'offline' && styles.resultWarn,
          ]}>
            <Text style={styles.resultText}>{lastResult.message}</Text>
          </View>
        )}

        {/* Actions */}
        <View style={styles.actionsCard}>
          <Text style={styles.actionsSectionTitle}>DEVICE MANAGEMENT</Text>
          <TouchableOpacity style={styles.actionRow} onPress={handlePurgeSynced}>
            <Text style={styles.actionIcon}>🧹</Text>
            <View style={styles.actionInfo}>
              <Text style={styles.actionLabel}>Purge synced records</Text>
              <Text style={styles.actionSub}>{stats.syncedLocal} record(s) uploaded and safe to remove</Text>
            </View>
            <Text style={styles.actionChevron}>›</Text>
          </TouchableOpacity>
          <View style={styles.divider} />
          <TouchableOpacity style={styles.actionRow} onPress={refresh}>
            <Text style={styles.actionIcon}>↻</Text>
            <View style={styles.actionInfo}>
              <Text style={styles.actionLabel}>Refresh stats</Text>
            </View>
            <Text style={styles.actionChevron}>›</Text>
          </TouchableOpacity>
        </View>

        {/* Danger */}
        <TouchableOpacity style={styles.dangerBtn} onPress={handlePurgeAll}>
          <Text style={styles.dangerBtnText}>🗑  Purge All Local Data</Text>
        </TouchableOpacity>

      </ScrollView>
    </SafeAreaView>
  );
};

const StatCard: React.FC<{ label: string; value: number; color: string; icon: string }> = ({ label, value, color, icon }) => (
  <View style={[styles.statCard, { borderTopColor: color }]}>
    <Text style={styles.statIcon}>{icon}</Text>
    <Text style={[styles.statValue, { color }]}>{value}</Text>
    <Text style={styles.statLabel}>{label}</Text>
  </View>
);

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0f172a' },
  scroll: { flexGrow: 1, padding: 20 },

  brand: { flexDirection: 'row', alignItems: 'center', gap: 14, marginBottom: 20, marginTop: 8 },
  brandBadge: { backgroundColor: '#1d4ed8', borderRadius: 10, paddingHorizontal: 12, paddingVertical: 8 },
  brandBadgeText: { color: '#fff', fontWeight: '900', fontSize: 16, letterSpacing: 1.5 },
  brandTitle: { fontSize: 22, fontWeight: '700', color: '#f8fafc' },
  brandSub: { fontSize: 13, color: '#64748b', marginTop: 2 },

  netBanner: {
    flexDirection: 'row', alignItems: 'center', gap: 10,
    borderRadius: 12, padding: 14, marginBottom: 20, borderWidth: 1,
  },
  netOnline: { backgroundColor: '#052e16', borderColor: '#166534' },
  netOffline: { backgroundColor: '#1c1917', borderColor: '#44403c' },
  netDot: { width: 9, height: 9, borderRadius: 5 },
  dotGreen: { backgroundColor: '#22c55e' },
  dotRed: { backgroundColor: '#ef4444' },
  dotGrey: { backgroundColor: '#64748b' },
  netText: { color: '#cbd5e1', fontSize: 13, flex: 1 },

  statsRow: { flexDirection: 'row', gap: 10, marginBottom: 20 },
  statCard: {
    flex: 1, backgroundColor: '#1e293b', borderRadius: 14, padding: 14,
    alignItems: 'center', borderTopWidth: 3,
  },
  statIcon: { fontSize: 18, marginBottom: 6 },
  statValue: { fontSize: 26, fontWeight: '800', marginBottom: 2 },
  statLabel: { fontSize: 10, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5 },

  card: {
    backgroundColor: '#1e293b', borderRadius: 16, padding: 18,
    marginBottom: 16, borderWidth: 1, borderColor: '#334155',
  },
  cardHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 },
  cardTitle: { fontSize: 16, fontWeight: '700', color: '#f8fafc' },
  badge: { borderRadius: 20, paddingHorizontal: 10, paddingVertical: 4 },
  badgeText: { fontSize: 12, fontWeight: '700' },
  cardBody: { color: '#94a3b8', fontSize: 13, lineHeight: 20, marginBottom: 8 },
  cardHint: { color: '#475569', fontSize: 11 },

  resultBox: {
    backgroundColor: '#052e16', borderRadius: 10, padding: 12,
    marginBottom: 16, borderLeftWidth: 4, borderLeftColor: '#22c55e',
  },
  resultWarn: { backgroundColor: '#1c1400', borderLeftColor: '#f59e0b' },
  resultError: { backgroundColor: '#450a0a', borderLeftColor: '#ef4444' },
  resultText: { color: '#e2e8f0', fontSize: 13, lineHeight: 18 },

  actionsCard: {
    backgroundColor: '#1e293b', borderRadius: 16, overflow: 'hidden',
    marginBottom: 20, borderWidth: 1, borderColor: '#334155',
  },
  actionsSectionTitle: { fontSize: 11, fontWeight: '700', color: '#475569', padding: 14, paddingBottom: 8, letterSpacing: 1 },
  actionRow: { flexDirection: 'row', alignItems: 'center', padding: 16, gap: 14 },
  actionIcon: { fontSize: 20 },
  actionInfo: { flex: 1 },
  actionLabel: { fontSize: 15, fontWeight: '600', color: '#f8fafc' },
  actionSub: { fontSize: 12, color: '#64748b', marginTop: 2 },
  actionChevron: { color: '#475569', fontSize: 20, fontWeight: '300' },
  divider: { height: 1, backgroundColor: '#334155', marginLeft: 52 },

  dangerBtn: {
    borderWidth: 1, borderColor: '#7f1d1d', borderRadius: 14,
    padding: 16, alignItems: 'center', backgroundColor: '#450a0a',
  },
  dangerBtnText: { color: '#fca5a5', fontSize: 15, fontWeight: '700' },
});

export default AdminSyncScreen;
