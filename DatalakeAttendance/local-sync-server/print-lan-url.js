/**
 * Prints the URL to paste into src/config/syncConfig.ts (physical device on WiFi).
 */
const os = require('os');
const PORT = Number(process.env.PORT) || 8787;

const nets = os.networkInterfaces();
const ips = [];

for (const name of Object.keys(nets)) {
  for (const net of nets[name] || []) {
    if (net.family === 'IPv4' && !net.internal) {
      ips.push({ iface: name, address: net.address });
    }
  }
}

console.log('\nPaste into src/config/syncConfig.ts → LOCAL_SYNC_HOST:\n');
if (ips.length === 0) {
  console.log('  (no LAN IPv4 found — check WiFi / Ethernet)\n');
} else {
  for (const { iface, address } of ips) {
    console.log(`  // ${iface}`);
    console.log(`  http://${address}:${PORT}/attendance\n`);
  }
}
console.log('Android emulator use: http://10.0.2.2:' + PORT + '/attendance\n');
