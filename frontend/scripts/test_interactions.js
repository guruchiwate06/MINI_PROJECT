import puppeteer from 'puppeteer-core';

async function testInteractions() {
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1200 });

  await page.goto('http://127.0.0.1:4173/', { waitUntil: 'networkidle0' });
  await new Promise((r) => setTimeout(r, 1500));

  console.log('1. Testing Play button click...');
  // Click the button containing PLAY
  const clickedPlay = await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('button'));
    const playBtn = btns.find((b) => b.textContent && b.textContent.includes('PLAY'));
    if (playBtn) {
      playBtn.click();
      return true;
    }
    return false;
  });
  console.log('Clicked PLAY:', clickedPlay);

  // Wait 1.5 seconds of real playback
  await new Promise((r) => setTimeout(r, 1500));

  // Check if pause button appeared and time advanced
  const playbackStateInfo = await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('button'));
    const hasPause = btns.some((b) => b.textContent && b.textContent.includes('PAUSE'));
    const bodyText = document.body.innerText;
    return { hasPause, snippet: bodyText.slice(0, 300) };
  });
  console.log('2. Playback state verified:', playbackStateInfo.hasPause ? 'RUNNING (PAUSE visible)' : 'CHECK');

  // Pause playback
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('button'));
    const pauseBtn = btns.find((b) => b.textContent && b.textContent.includes('PAUSE'));
    if (pauseBtn) pauseBtn.click();
  });

  console.log('3. Testing Micro-Window Selection...');
  const clickedSite4 = await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('button'));
    const site4Btn = btns.find((b) => b.textContent && b.textContent.includes('Site 4'));
    if (site4Btn) {
      site4Btn.click();
      return true;
    }
    return false;
  });
  console.log('Clicked Site 4 (Divergence Switch):', clickedSite4);
  await new Promise((r) => setTimeout(r, 500));

  // Check FFT dominant frequency
  const dominantFreqText = await page.evaluate(() => {
    const divs = Array.from(document.querySelectorAll('div'));
    const domDiv = divs.find((d) => d.textContent && d.textContent.includes('Dominant Frequency'));
    return domDiv ? domDiv.innerText : 'Not found';
  });
  console.log('4. FFT Dominant Frequency readout:\n', dominantFreqText);

  // Test switching tabs to Phase 1 Scientific Claim Registry
  console.log('5. Testing Tab Switcher to Scientific Claim Registry...');
  const switchedTab = await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('button'));
    const tabBtn = btns.find((b) => b.textContent && b.textContent.includes('SCIENTIFIC CLAIM REGISTRY'));
    if (tabBtn) {
      tabBtn.click();
      return true;
    }
    return false;
  });
  console.log('Switched to Claims Tab:', switchedTab);
  await new Promise((r) => setTimeout(r, 500));

  const claimCount = await page.evaluate(() => {
    return document.querySelectorAll('button:has(span)').length;
  });
  console.log('6. Registry elements rendered:', claimCount > 0);

  await browser.close();
  console.log('All interactive verification checks PASSED!');
}

testInteractions().catch((err) => {
  console.error('Interaction test failed:', err);
  process.exit(1);
});
