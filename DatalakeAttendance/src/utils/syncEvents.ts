import type { SyncResult } from './attendanceSync';

export type SyncEvent =
  | { type: 'started' }
  | { type: 'finished'; result: SyncResult };

type Listener = (event: SyncEvent) => void;
const listeners = new Set<Listener>();

export function subscribeSyncEvents(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function emitSyncEvent(event: SyncEvent): void {
  listeners.forEach(l => {
    try {
      l(event);
    } catch {
      // ignore listener errors
    }
  });
}
