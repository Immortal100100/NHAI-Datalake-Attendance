/**
 * Local attendance sync server for Phase 7 testing.
 * Listens on all interfaces so a phone on the same WiFi can POST.
 *
 *   node server.js
 *   GET  http://localhost:8787/health
 *   POST http://localhost:8787/attendance
 *   GET  http://localhost:8787/records
 */

const http = require('http');

const PORT = Number(process.env.PORT) || 8787;
const API_KEY = process.env.SYNC_API_KEY || 'dev-local-key';

/** @type {Array<{ receivedAt: string; payload: object }>} */
const store = [];

function json(res, status, body) {
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  });
  res.end(JSON.stringify(body, null, 2));
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', c => chunks.push(c));
    req.on('end', () => {
      try {
        const raw = Buffer.concat(chunks).toString('utf8');
        resolve(raw ? JSON.parse(raw) : {});
      } catch (e) {
        reject(e);
      }
    });
    req.on('error', reject);
  });
}

function checkAuth(req) {
  const auth = req.headers.authorization || '';
  if (!API_KEY) return true;
  return auth === `Bearer ${API_KEY}`;
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url || '/', `http://${req.headers.host || 'localhost'}`);

  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    });
    res.end();
    return;
  }

  if (url.pathname === '/health' && req.method === 'GET') {
    json(res, 200, {
      ok: true,
      service: 'datalake-local-sync',
      port: PORT,
      storedBatches: store.length,
    });
    return;
  }

  if (url.pathname === '/records' && req.method === 'GET') {
    json(res, 200, { batches: store });
    return;
  }

  if (url.pathname === '/attendance' && req.method === 'POST') {
    if (!checkAuth(req)) {
      json(res, 401, { ok: false, error: 'Invalid or missing Authorization Bearer token' });
      return;
    }
    try {
      const payload = await readBody(req);
      const records = Array.isArray(payload.records) ? payload.records : [];
      if (records.length === 0) {
        json(res, 400, { ok: false, error: 'Expected payload.records array with at least one item' });
        return;
      }
      store.push({ receivedAt: new Date().toISOString(), payload });
      console.log(`[sync] +${records.length} record(s) from ${payload.source || 'unknown'}`);
      records.forEach((r, i) => {
        const loc =
          r.latitude != null && r.longitude != null
            ? ` lat=${r.latitude.toFixed(5)} lng=${r.longitude.toFixed(5)} acc=${r.locationAccuracy ?? '?'}m`
            : ' lat=— lng=— (GPS missing — enable Location permission on phone)';
        console.log(
          `  [${i + 1}] id=${r.id} user=${r.userId} name=${r.name} time=${r.checkInTime} score=${r.similarityScore}${loc}`,
        );
      });
      json(res, 200, {
        ok: true,
        accepted: records.length,
        batchId: store.length,
        message: 'Attendance batch stored locally',
      });
    } catch (e) {
      console.error('[sync] parse error', e);
      json(res, 400, { ok: false, error: 'Invalid JSON body' });
    }
    return;
  }

  json(res, 404, { ok: false, error: 'Not found', paths: ['/health', '/attendance', '/records'] });
});

server.listen(PORT, '0.0.0.0', () => {
  console.log('');
  console.log('  Datalake local sync server');
  console.log('  --------------------------');
  console.log(`  Listening on 0.0.0.0:${PORT}`);
  console.log(`  API key (Bearer): ${API_KEY}`);
  console.log('');
  console.log('  Emulator app URL:  http://10.0.2.2:' + PORT + '/attendance');
  console.log('  Physical device:   http://<YOUR_PC_LAN_IP>:' + PORT + '/attendance');
  console.log('  Run:  npm run url   (prints LAN URL for syncConfig.ts)');
  console.log('');
});
