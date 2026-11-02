import puppeteer from 'puppeteer-core';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

async function capture() {
  console.log('Starting preview server on port 4173...');
  const server = spawn('npm.cmd', ['run', 'preview', '--', '--port', '4173'], {
    cwd: process.cwd(),
    shell: true,
  });

  // Give server 3 seconds to spin up
  await new Promise(r => setTimeout(r, 3000));

  console.log('Launching Edge headless browser...');
  const edgePath = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
  const browser = await puppeteer.launch({
    executablePath: edgePath,
    headless: true,
    userDataDir: path.resolve('.puppeteer_profile'),
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 1080 });
    console.log('Navigating to http://localhost:4173 ...');
    await page.goto('http://localhost:4173', { waitUntil: 'networkidle0', timeout: 15000 });

    const outDir = path.resolve('public');
    const outPath1 = path.join(outDir, 'phase1_screenshot.png');
    const outPath2 = 'C:\\Users\\Thrill3r\\.gemini\antigravity\\brain\\776dc837-85e2-4c94-82fd-0b036fa4dd07\\phase1_screenshot.png';

    await page.screenshot({ path: outPath1, fullPage: false });
    console.log('Saved screenshot to:', outPath1);

    try {
      fs.copyFileSync(outPath1, outPath2);
      console.log('Copied screenshot to artifact directory:', outPath2);
    } catch (e) {
      console.warn('Could not copy to artifact dir:', e.message);
    }

    const title = await page.title();
    console.log('Page Title:', title);
    const textContent = await page.$eval('h1', el => el.innerText);
    console.log('Header text:', textContent);

  } catch (err) {
    console.error('Capture error:', err);
  } finally {
    await browser.close();
    server.kill();
    process.exit(0);
  }
}

capture();
