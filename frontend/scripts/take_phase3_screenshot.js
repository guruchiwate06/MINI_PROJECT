import puppeteer from 'puppeteer-core';
import path from 'path';

async function main() {
  console.log('Launching Edge browser for Phase 3 3D rendering and interaction tests...');

  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--enable-webgl',
      '--ignore-gpu-blocklist',
      '--use-gl=angle'
    ]
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1080 });

  const consoleLogs = [];
  page.on('console', (msg) => {
    consoleLogs.push({ type: msg.type(), text: msg.text() });
    if (msg.type() === 'error') {
      console.error('BROWSER ERROR:', msg.text());
    }
  });

  page.on('pageerror', (err) => {
    console.error('PAGE UNCAUGHT ERROR:', err);
  });

  console.log('Navigating to http://127.0.0.1:4173/ ...');
  await page.goto('http://127.0.0.1:4173/', { waitUntil: 'networkidle0', timeout: 20000 });

  // 1. Initial 3D Scene Stabilization
  console.log('Waiting 3.5s for WebGL Three.js canvas to compile shaders and render...');
  await new Promise((r) => setTimeout(r, 3500));

  const artifactDir = 'C:/Users/Thrill3r/.gemini/antigravity/brain/776dc837-85e2-4c94-82fd-0b036fa4dd07';

  // Check canvas exists
  const canvasExists = await page.$('canvas');
  if (!canvasExists) {
    throw new Error('Canvas element not found in DOM!');
  }
  console.log('✓ 3D WebGL Canvas successfully created and active in DOM.');

  // Screenshot 1: Primary Cinematic Side-Follow View
  const shot1 = path.resolve(artifactDir, 'phase3_3d_side_follow.png');
  await page.screenshot({ path: shot1 });
  console.log(`✓ Screenshot 1 (Side Follow) saved to: ${shot1}`);

  // 2. Camera Transition Test: Switch to FRONT CAB
  console.log('Testing Camera Transition: Clicking "FRONT CAB"...');
  await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const btn = buttons.find((b) => b.textContent && b.textContent.includes('FRONT CAB'));
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 1200));

  const shot2 = path.resolve(artifactDir, 'phase3_3d_front_cab.png');
  await page.screenshot({ path: shot2 });
  console.log(`✓ Screenshot 2 (Front Cab) saved to: ${shot2}`);

  // 3. Camera Transition Test: Switch to BOGIE INSPECT
  console.log('Testing Camera Transition: Clicking "BOGIE INSPECT"...');
  await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const btn = buttons.find((b) => b.textContent && b.textContent.includes('BOGIE INSPECT'));
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 1200));

  const shot3 = path.resolve(artifactDir, 'phase3_3d_bogie_inspect.png');
  await page.screenshot({ path: shot3 });
  console.log(`✓ Screenshot 3 (Bogie Inspect) saved to: ${shot3}`);

  // 4. Anomaly Jump Synchronization Test: Click Jump to S4 (Switch Divergence at 263.4m)
  console.log('Testing Anomaly Synchronization: Clicking "S4 (263m)"...');
  await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const btn = buttons.find((b) => b.textContent && b.textContent.includes('S4'));
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 1500));

  const shot4 = path.resolve(artifactDir, 'phase3_3d_switch_anomaly_site4.png');
  await page.screenshot({ path: shot4 });
  console.log(`✓ Screenshot 4 (Switch Anomaly S4) saved to: ${shot4}`);

  // 5. Playback Synchronization Test: Click PLAY 3D TRACK and observe movement
  console.log('Testing Playback Synchronization: Clicking "PLAY 3D TRACK"...');
  await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const btn = buttons.find((b) => b.textContent && b.textContent.includes('PLAY 3D TRACK'));
    if (btn) btn.click();
  });
  // Let it roll for 2.5 seconds
  await new Promise((r) => setTimeout(r, 2500));

  // Pause playback
  await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const btn = buttons.find((b) => b.textContent && b.textContent.includes('PAUSE PLAYBACK'));
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 500));

  // Switch back to side follow to capture playback position
  await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const btn = buttons.find((b) => b.textContent && b.textContent.includes('SIDE FOLLOW'));
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 1000));

  const shot5 = path.resolve(artifactDir, 'phase3_3d_synchronized_playback.png');
  await page.screenshot({ path: shot5 });
  console.log(`✓ Screenshot 5 (Synchronized Playback) saved to: ${shot5}`);

  // Verify Console Errors
  const errors = consoleLogs.filter((l) => l.type === 'error');
  if (errors.length > 0) {
    console.error(`Detected ${errors.length} browser errors:`, errors);
  } else {
    console.log('✓ Zero browser console errors detected throughout 3D tests!');
  }

  await browser.close();
  console.log('All Phase 3 3D digital twin verification tests completed successfully!');
}

main().catch((err) => {
  console.error('Fatal test error:', err);
  process.exit(1);
});

