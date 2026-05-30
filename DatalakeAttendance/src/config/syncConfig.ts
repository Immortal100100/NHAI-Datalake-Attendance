/**
 * Sync endpoint configuration.
 *
 * Dummy AWS (current): webhook.site — view POSTs at link below.
 * Local test server:   cd local-sync-server && npm start
 * Production:          Set USE_LOCAL_SYNC_SERVER = false, set AWS_SYNC_URL
 */

/** Set false to use AWS_SYNC_URL below. */
export const USE_LOCAL_SYNC_SERVER = false;

/** Local dev server (same WiFi as PC). */
export const LOCAL_SYNC_HOST = 'http://192.168.29.139:8787/attendance';

/**
 * Dummy AWS endpoint — powered by webhook.site.
 * View received POSTs at:
 *   https://webhook.site/#!/5678eed5-7875-466a-a724-dd3761243de4
 */
export const AWS_SYNC_URL = 'https://webhook.site/5678eed5-7875-466a-a724-dd3761243de4';

export const SYNC_CONFIG = {
  apiUrl: USE_LOCAL_SYNC_SERVER ? LOCAL_SYNC_HOST : AWS_SYNC_URL,

  /** API key sent as Authorization Bearer. webhook.site ignores it (fine for testing). */
  apiKey: '',

  /** Only used when apiUrl is empty. */
  demoModeWhenNoUrl: false,

  timeoutMs: 15000,
} as const;

export function isSyncEndpointConfigured(): boolean {
  return SYNC_CONFIG.apiUrl.trim().length > 0;
}
