'use strict';
// Generates 24+24 placeholder SVGs (NOT faces) so the app can be tried end-to-end. Replace with real stimuli.
const fs = require('fs'), path = require('path');
for (const cls of ['real', 'ai']) {
  const dir = path.join(__dirname, '..', 'stimuli', cls);
  fs.mkdirSync(dir, { recursive: true });
  for (let i = 1; i <= 24; i++) {
    const hue = (i * 37 + (cls === 'ai' ? 11 : 0)) % 360;
    fs.writeFileSync(path.join(dir, `demo_${cls}_${String(i).padStart(2, '0')}.svg`),
      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400"><rect width="400" height="400" fill="hsl(${hue},25%,70%)"/>` +
      `<circle cx="200" cy="170" r="95" fill="hsl(${hue},35%,86%)"/><circle cx="165" cy="150" r="9"/><circle cx="235" cy="150" r="9"/>` +
      `<path d="M155 205 Q200 240 245 205" stroke="#000" stroke-width="6" fill="none"/>` +
      `<text x="200" y="350" font-size="26" text-anchor="middle" fill="#222" font-family="sans-serif">DEMO ${i}</text></svg>`);
  }
}
console.log('Demo stimuli written to stimuli/real and stimuli/ai');
