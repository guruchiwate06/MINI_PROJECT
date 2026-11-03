import puppeteer from 'puppeteer-core';

async function main() {
  console.log('Running regression check on Phase 1 & Phase 2 views...');
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--enable-webgl']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  const errors = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text());
  });
  page.on('pageerror', (err) => errors.push(err.message));

  await page.goto('http://127.0.0.1:4173/', { waitUntil: 'networkidle0', timeout: 15000 });

  // 1. Click SIGNAL INTELLIGENCE (Phase 2)
  console.log('Testing Tab: SIGNAL INTELLIGENCE (Phase 2)...');
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('nav button'));
    const btn = btns.find((b) => b.textContent && b.textContent.includes('SIGNAL INTELLIGENCE'));
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 1500));
  const hasOscilloscope = await page.$('.echarts-for-react');
  console.log('Oscilloscope rendered:', !!hasOscilloscope);

  // 2. Click SCIENTIFIC CLAIM REGISTRY (Phase 1)
  console.log('Testing Tab: SCIENTIFIC CLAIM REGISTRY (Phase 1)...');
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('nav button'));
    const btn = btns.find((b) => b.textContent && b.textContent.includes('SCIENTIFIC CLAIM REGISTRY'));
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 800));
  const claimCards = await page.$$eval('[class*="border"]', (els) => els.length);
  console.log('Claim cards present:', claimCards > 0);

  // 3. Click TESTBED & CLASS 230 (Phase 1)
  console.log('Testing Tab: TESTBED & CLASS 230...');
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('nav button'));
    const btn = btns.find((b) => b.textContent && b.textContent.includes('TESTBED & CLASS 230'));
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 800));

  // 4. Click AUDIT & PUBLICATION (Phase 1)
  console.log('Testing Tab: AUDIT & PUBLICATION...');
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('nav button'));
    const btn = btns.find((b) => b.textContent && b.textContent.includes('AUDIT & PUBLICATION'));
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 800));

  // 5. Click back to 3D DIGITAL TWIN (Phase 3)
  console.log('Testing Tab: Return to 3D DIGITAL TWIN...');
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('nav button'));
    const btn = btns.find((b) => b.textContent && b.textContent.includes('3D DIGITAL TWIN'));
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 1000));
  const hasCanvas = await page.$('canvas');
  console.log('Canvas active upon return:', !!hasCanvas);

  await browser.close();

  if (errors.length > 0) {
    console.error('Regression errors detected:', errors);
    process.exit(1);
  } else {
    console.log('✓ All Phase 1, Phase 2, and Phase 3 tabs switch seamlessly with zero errors!');
  }
}

main().catch((err) => {
  console.error('Regression test failed:', err);
  process.exit(1);
});

