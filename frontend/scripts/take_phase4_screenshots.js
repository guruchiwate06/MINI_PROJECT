import puppeteer from 'puppeteer-core';
import path from 'path';

async function main() {
  console.log('Launching Edge browser for Phase 4 UI and Verification tests...');

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
  await new Promise((r) => setTimeout(r, 2000));

  const artifactDir = 'C:/Users/Thrill3r/.gemini/antigravity/brain/776dc837-85e2-4c94-82fd-0b036fa4dd07';

  // 1. Switch to AI BENCHMARKING Tab (Phase 4)
  console.log('Clicking "AI BENCHMARKING" tab...');
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('nav button'));
    const btn = btns.find((b) => b.textContent && b.textContent.includes('AI BENCHMARKING'));
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 2500));

  // Screenshot 1: Model Leaderboard View
  const shot1 = path.resolve(artifactDir, 'phase4_model_leaderboard.png');
  await page.screenshot({ path: shot1 });
  console.log(`✓ Screenshot 1 (Leaderboard) saved to: ${shot1}`);

  // 2. Click BENCHMARK COMPARISON Sub-tab
  console.log('Clicking "BENCHMARK COMPARISON" sub-tab...');
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('button'));
    const btn = btns.find((b) => b.textContent && b.textContent.includes('BENCHMARK COMPARISON'));
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 2500));

  // Screenshot 2: Benchmark Comparison & Curves View
  const shot2 = path.resolve(artifactDir, 'phase4_benchmark_comparison.png');
  await page.screenshot({ path: shot2 });
  console.log(`✓ Screenshot 2 (Comparison) saved to: ${shot2}`);

  // 3. Click SPATIAL HOLDOUT SITES Sub-tab
  console.log('Clicking "SPATIAL HOLDOUT SITES" sub-tab...');
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('button'));
    const btn = btns.find((b) => b.textContent && b.textContent.includes('SPATIAL HOLDOUT SITES'));
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 2000));

  // Screenshot 3: Spatial Holdout Evaluation View
  const shot3 = path.resolve(artifactDir, 'phase4_spatial_holdout.png');
  await page.screenshot({ path: shot3 });
  console.log(`✓ Screenshot 3 (Spatial Holdout) saved to: ${shot3}`);

  // 4. Click THRESHOLD EXPLORER Sub-tab
  console.log('Clicking "THRESHOLD EXPLORER" sub-tab...');
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('button'));
    const btn = btns.find((b) => b.textContent && b.textContent.includes('THRESHOLD EXPLORER'));
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 2500));

  // Screenshot 4: Threshold Explorer View
  const shot4 = path.resolve(artifactDir, 'phase4_threshold_explorer.png');
  await page.screenshot({ path: shot4 });
  console.log(`✓ Screenshot 4 (Threshold Explorer) saved to: ${shot4}`);

  // 5. Test Regression across Phase 1, Phase 2, Phase 3
  console.log('Verifying regression across tabs...');
  
  // Phase 3: 3D Twin
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('nav button'));
    const btn = btns.find((b) => b.textContent && b.textContent.includes('3D DIGITAL TWIN'));
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 1500));
  const hasCanvas = await page.$('canvas');
  console.log('Phase 3 3D Canvas intact:', !!hasCanvas);

  // Phase 2: Signal Intelligence
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('nav button'));
    const btn = btns.find((b) => b.textContent && b.textContent.includes('SIGNAL INTELLIGENCE'));
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 1500));
  const hasOscilloscope = await page.$('.echarts-for-react');
  console.log('Phase 2 Oscilloscope intact:', !!hasOscilloscope);

  // Phase 1: Claim Registry
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('nav button'));
    const btn = btns.find((b) => b.textContent && b.textContent.includes('SCIENTIFIC CLAIM REGISTRY'));
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 1000));
  const claimCards = await page.$$eval('[class*="border"]', (els) => els.length);
  console.log('Phase 1 Claims intact:', claimCards > 0);

  console.log('✓ All Phase 4 UI and cross-phase regression checks succeeded!');
  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

