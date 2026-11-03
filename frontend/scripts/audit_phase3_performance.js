import puppeteer from 'puppeteer-core';

async function auditPerformance() {
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--enable-webgl',
      '--ignore-gpu-blocklist',
      '--use-gl=angle',
      '--enable-features=VaapiVideoDecoder'
    ]
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1080 });

  await page.goto('http://127.0.0.1:4173/', { waitUntil: 'networkidle0', timeout: 20000 });
  await new Promise((r) => setTimeout(r, 3000));

  const perfData = await page.evaluate(async () => {
    // 1. Measure frame times over 120 frames
    const frameTimes = [];
    await new Promise((resolve) => {
      let count = 0;
      function step(now) {
        frameTimes.push(now);
        count++;
        if (count < 120) {
          requestAnimationFrame(step);
        } else {
          resolve();
        }
      }
      requestAnimationFrame(step);
    });

    const intervals = [];
    for (let i = 1; i < frameTimes.length; i++) {
      intervals.push(frameTimes[i] - frameTimes[i - 1]);
    }
    const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const empiricalFps = 1000 / avgInterval;

    // 2. Query WebGL context & extensions
    const canvas = document.querySelector('canvas');
    let glInfo = {};
    if (canvas) {
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      if (gl) {
        const debug = gl.getExtension('WEBGL_debug_renderer_info');
        glInfo = {
          renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : 'Unknown',
          vendor: debug ? gl.getParameter(debug.UNMASKED_VENDOR_WEBGL) : 'Unknown',
          drawingBufferWidth: gl.drawingBufferWidth,
          drawingBufferHeight: gl.drawingBufferHeight,
          version: gl.getParameter(gl.VERSION),
          shadingLanguageVersion: gl.getParameter(gl.SHADING_LANGUAGE_VERSION)
        };
      }
    }

    // 3. Inspect Memory if available
    const memory = (window.performance && window.performance.memory) ? {
      totalJSHeapMB: (window.performance.memory.totalJSHeapSize / (1024 * 1024)).toFixed(2),
      usedJSHeapMB: (window.performance.memory.usedJSHeapSize / (1024 * 1024)).toFixed(2),
      jsHeapSizeLimitMB: (window.performance.memory.jsHeapSizeLimit / (1024 * 1024)).toFixed(2)
    } : null;

    return {
      empiricalFps: Math.round(empiricalFps * 100) / 100,
      avgFrameTimeMs: Math.round(avgInterval * 100) / 100,
      minFrameTimeMs: Math.min(...intervals).toFixed(2),
      maxFrameTimeMs: Math.max(...intervals).toFixed(2),
      glInfo,
      memory
    };
  });

  console.log('AUDIT_PERFORMANCE_RESULT:');
  console.log(JSON.stringify(perfData, null, 2));

  await browser.close();
}

auditPerformance().catch((err) => {
  console.error(err);
  process.exit(1);
});
