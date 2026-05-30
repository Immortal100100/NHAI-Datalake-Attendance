import { PermissionsAndroid, Platform } from 'react-native';
import Geolocation, {
  type GeoError,
  type GeoPosition,
} from 'react-native-geolocation-service';

/** Max wait for a GPS fix during check-in (Phase 6 spec). */
export const LOCATION_TIMEOUT_MS = 4000;

/** Reuse last fix if younger than this (Phase 6 spec). */
export const LOCATION_CACHE_MS = 60_000;

export interface LocationFix {
  latitude: number;
  longitude: number;
  accuracy: number;
  /** True when returned from in-memory or OS cache without a fresh fix. */
  cached: boolean;
  timestamp: number;
}

let memoryCache: LocationFix | null = null;

function isFresh(fix: LocationFix): boolean {
  return Date.now() - fix.timestamp < LOCATION_CACHE_MS;
}

export async function requestLocationPermission(): Promise<boolean> {
  try {
    if (Platform.OS === 'ios') {
      const status = await Geolocation.requestAuthorization('whenInUse');
      return status === 'granted';
    }
    const result = await PermissionsAndroid.request(
      PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
      {
        title: 'Location access',
        message:
          'DatalakeAttendance records GPS coordinates with attendance for field verification.',
        buttonNeutral: 'Ask later',
        buttonNegative: 'Cancel',
        buttonPositive: 'Allow',
      },
    );
    return result === PermissionsAndroid.RESULTS.GRANTED;
  } catch {
    return false;
  }
}

function positionToFix(position: GeoPosition, cached: boolean): LocationFix {
  return {
    latitude: position.coords.latitude,
    longitude: position.coords.longitude,
    accuracy: position.coords.accuracy ?? 0,
    cached,
    timestamp: position.timestamp ?? Date.now(),
  };
}

function fetchCurrentPosition(): Promise<LocationFix> {
  return new Promise((resolve, reject) => {
    Geolocation.getCurrentPosition(
      position => resolve(positionToFix(position, false)),
      (error: GeoError) => reject(new Error(error.message || `GPS error ${error.code}`)),
      {
        enableHighAccuracy: true,
        timeout: LOCATION_TIMEOUT_MS,
        maximumAge: LOCATION_CACHE_MS,
        forceRequestLocation: false,
        showLocationDialog: true,
      },
    );
  });
}

/**
 * Returns GPS for attendance logging. Uses a 60s in-memory cache and
 * 4s timeout per fix. Works offline (GPS does not need internet).
 */
export async function getAttendanceLocation(): Promise<LocationFix | null> {
  if (memoryCache && isFresh(memoryCache)) {
    return { ...memoryCache, cached: true };
  }

  const granted = await requestLocationPermission();
  if (!granted) {
    return memoryCache && isFresh(memoryCache) ? { ...memoryCache, cached: true } : null;
  }

  try {
    const fix = await fetchCurrentPosition();
    memoryCache = fix;
    return fix;
  } catch {
    if (memoryCache && isFresh(memoryCache)) {
      return { ...memoryCache, cached: true };
    }
    return null;
  }
}

/** Fire-and-forget cache warm before face capture starts. */
export function warmLocationCache(): void {
  getAttendanceLocation().catch(() => {});
}

export function formatLocationSummary(fix: LocationFix | null): string {
  if (!fix) {
    return 'GPS unavailable';
  }
  const tag = fix.cached ? 'cached' : 'live';
  return `${fix.latitude.toFixed(5)}, ${fix.longitude.toFixed(5)} (±${Math.round(fix.accuracy)}m, ${tag})`;
}
