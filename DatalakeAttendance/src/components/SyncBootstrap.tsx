import React from 'react';
import { useAttendanceAutoSync } from '../hooks/useAttendanceAutoSync';

/** Mount once at app root — background sync on WiFi connect / app open. */
const SyncBootstrap: React.FC = () => {
  useAttendanceAutoSync();
  return null;
};

export default SyncBootstrap;
