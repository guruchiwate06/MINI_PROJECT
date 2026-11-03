import puppeteer from 'puppeteer-core';
import path from 'path';
import fs from 'fs';

async function capture() {
  console.log('Launching Edge headless browser...');
  const edgePath = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
  const browser = await puppeteer.launch({
    executablePath: edgePath,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1560, height: 1120 });
    console.log('Navigating to http://127.0.0.1:4173 ...');
    await page.goto('http://127.0.0.1:4173', { waitUntil: 'domcontentloaded', timeout: 15000 });
    await new Promise(r => setTimeout(r, 2500));

    // Click on AI Benchmarking (Phase 4 tab)
    console.log('Navigating to Phase 4 (AI Benchmarking)...');
    const buttons = await page.$$('button');
    for (const b of buttons) {
      const text = await page.evaluate(el => el.innerText, b);
      if (text && text.includes('AI BENCHMARKING')) {
        await b.click();
        break;
      }
    }
    await new Promise(r => setTimeout(r, 2000));

    const outDir1 = path.resolve('model_training/reports/phase4_audit_screenshots');
    const outDir2 = 'C:\\Users\\Thrill3r\\.gemini\\antigravity\\brain\\c7e88844-ca13-4f64-98f2-54421df8d069';
    fs.mkdirSync(outDir1, { recursive: true });

    // 1. Screenshot 1: Test-evaluated model leaderboard
    console.log('Capturing Screenshot 1: Test-evaluated model leaderboard...');
    const s1Path1 = path.join(outDir1, '01_test_evaluated_leaderboard.png');
    const s1Path2 = path.join(outDir2, '01_test_evaluated_leaderboard.png');
    await page.screenshot({ path: s1Path1, fullPage: false });
    try { fs.copyFileSync(s1Path1, s1Path2); } catch (e) { }
    console.log('Saved 1:', s1Path1);

    // 2. Screenshot 2: Track-OT++ separate validation section
    console.log('Capturing Screenshot 2: Separate Track-OT++ validation status...');
    await page.evaluate(() => {
      window.scrollBy(0, 720);
    });
    await new Promise(r => setTimeout(r, 1000));
    const s2Path1 = path.join(outDir1, '02_track_ot_validation_status.png');
    const s2Path2 = path.join(outDir2, '02_track_ot_validation_status.png');
    await page.screenshot({ path: s2Path1, fullPage: false });
    try { fs.copyFileSync(s2Path1, s2Path2); } catch (e) { }
    console.log('Saved 2:', s2Path1);

    // Scroll back up
    await page.evaluate(() => window.scrollTo(0, 0));
    await new Promise(r => setTimeout(r, 600));

    // 3. Screenshot 3: AP versus PR-AUC comparison
    console.log('Capturing Screenshot 3: AP vs PR-AUC comparison...');
    const subButtons1 = await page.$$('button');
    for (const b of subButtons1) {
      const text = await page.evaluate(el => el.innerText, b);
      if (text && text.includes('BENCHMARK COMPARISON')) {
        await b.click();
        break;
      }
    }
    await new Promise(r => setTimeout(r, 2000));
    const s3Path1 = path.join(outDir1, '03_ap_vs_prauc_comparison.png');
    const s3Path2 = path.join(outDir2, '03_ap_vs_prauc_comparison.png');
    await page.screenshot({ path: s3Path1, fullPage: false });
    try { fs.copyFileSync(s3Path1, s3Path2); } catch (e) { }
    console.log('Saved 3:', s3Path1);

    // 4. Screenshot 4: Nine-site evaluation
    console.log('Capturing Screenshot 4: Nine-site evaluation...');
    const subButtons2 = await page.$$('button');
    for (const b of subButtons2) {
      const text = await page.evaluate(el => el.innerText, b);
      if (text && text.includes('SPATIAL HOLDOUT SITES')) {
        await b.click();
        break;
      }
    }
    await new Promise(r => setTimeout(r, 2000));
    const s4Path1 = path.join(outDir1, '04_nine_site_evaluation.png');
    const s4Path2 = path.join(outDir2, '04_nine_site_evaluation.png');
    await page.screenshot({ path: s4Path1, fullPage: false });
    try { fs.copyFileSync(s4Path1, s4Path2); } catch (e) { }
    console.log('Saved 4:', s4Path1);

    // 5. Screenshot 5: Interactive threshold explorer
    console.log('Capturing Screenshot 5: Interactive threshold explorer...');
    const subButtons3 = await page.$$('button');
    for (const b of subButtons3) {
      const text = await page.evaluate(el => el.innerText, b);
      if (text && text.includes('THRESHOLD EXPLORER')) {
        await b.click();
        break;
      }
    }
    await new Promise(r => setTimeout(r, 2000));
    const s5Path1 = path.join(outDir1, '05_threshold_explorer.png');
    const s5Path2 = path.join(outDir2, '05_threshold_explorer.png');
    await page.screenshot({ path: s5Path1, fullPage: false });
    try { fs.copyFileSync(s5Path1, s5Path2); } catch (e) { }
    console.log('Saved 5:', s5Path1);

    console.log('SUCCESS: All 5 screenshots captured and saved to disk & brain artifact directory!');
  } catch (err) {
    console.error('Capture error:', err);
  } finally {
    await browser.close();
    process.exit(0);
  }
}

capture();
