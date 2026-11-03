import puppeteer from 'puppeteer-core';
import path from 'path';

async function main() {
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1800 });

  const consoleLogs = [];
  page.on('console', (msg) => {
    consoleLogs.push({ type: msg.type(), text: msg.text() });
  });

  page.on('pageerror', (err) => {
    console.error('PAGE ERROR:', err);
  });

  console.log('Navigating to http://127.0.0.1:4173/ ...');
  await page.goto('http://127.0.0.1:4173/', { waitUntil: 'networkidle0', timeout: 15000 });

  // Wait 2 seconds for ECharts and Zustand store to fully render
  await new Promise((r) => setTimeout(r, 2000));

  const artifactPath = path.resolve(
    'C:/Users/Thrill3r/.gemini/antigravity/brain/776dc837-85e2-4c94-82fd-0b036fa4dd07/phase2_screenshot.png'
  );
  await page.screenshot({ path: artifactPath, fullPage: true });
  console.log(`Screenshot saved to: ${artifactPath}`);

  console.log(`Captured ${consoleLogs.length} console messages.`);
  const errors = consoleLogs.filter((l) => l.type === 'error');
  if (errors.length > 0) {
    console.error('Browser errors detected:', errors);
  } else {
    console.log('Zero browser console errors detected!');
  }

  // Check DOM elements
  const title = await page.title();
  const headings = await page.$$eval('h1, h2, h3', (els) => els.map((e) => e.textContent.trim()));
  console.log('Page Title:', title);
  console.log('Headings found:', headings);

  await browser.close();
}

main().catch((err) => {
  console.error('Fatal test error:', err);
  process.exit(1);
});

