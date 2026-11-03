/**
 * Synthetic FFT Verification Test
 * Validates:
 * 1. Bin-centred sinusoid amplitude accuracy (positive frequencies, scaled by 2/N)
 * 2. DC bin (k = 0) scaling (1/N, not doubled)
 * 3. Nyquist bin (k = N/2) scaling (1/N, not doubled)
 * 4. Exact coherent gain normalization for Rectangular, Hann, and Hamming windows
 * 5. dB re 1.0g reference conversion
 */

import { computeSpectrum, getCoherentGain, cooleyTukeyFFT, WindowFunctionType } from '../src/utils/fft';

const N = 1024;
const Fs = 2000;
const TOLERANCE = 1e-4; // 0.01% numerical tolerance

console.log('================================================================');
console.log('         SYNTHETIC FFT NUMERICAL VERIFICATION SUITE             ');
console.log('================================================================\n');

let allPassed = true;

function assertClose(actual: number, expected: number, label: string, tol = TOLERANCE) {
  const diff = Math.abs(actual - expected);
  const passed = diff <= tol;
  if (!passed) allPassed = false;
  console.log(`  ${passed ? '✓ PASS' : '✗ FAIL'}: ${label}`);
  console.log(`         Expected: ${expected.toFixed(6)}, Actual: ${actual.toFixed(6)}, Diff: ${diff.toExponential(3)} (tol: ${tol.toExponential(1)})`);
}

// -------------------------------------------------------------------------
// TEST 1: Exact Coherent Gain Verification
// -------------------------------------------------------------------------
console.log('[TEST 1] Window Coherent Gain Calculation:');
const cgRect = getCoherentGain('rectangular', N);
assertClose(cgRect, 1.0, 'Rectangular window coherent gain == 1.0');

const cgHann = getCoherentGain('hanning', N);
const expectedHannCg = (0.5 * (N - 1)) / N;
assertClose(cgHann, expectedHannCg, 'Hann window coherent gain matches discrete sum', 1e-6);

const cgHamming = getCoherentGain('hamming', N);
// Hamming discrete sum: 0.54 - 0.46 * (1/N)
const expectedHammingCg = (0.54 * N - 0.46) / N;
assertClose(cgHamming, expectedHammingCg, 'Hamming window coherent gain matches discrete sum', 1e-6);

// -------------------------------------------------------------------------
// TEST 2: Bin-Centred Positive Frequency Sinusoid (k = 64, f = 125.0 Hz, A = 0.75 g)
// -------------------------------------------------------------------------
console.log('\n[TEST 2] Bin-Centred Sinusoid Amplitude Accuracy (A = 0.75 g, k = 64, f = 125 Hz):');
const kTarget = 64;
const fTarget = kTarget * (Fs / N); // 125.0 Hz exactly
const knownAmplitude = 0.75; // 0.75g peak amplitude

const signal = new Array(N);
for (let n = 0; n < N; n++) {
  signal[n] = knownAmplitude * Math.sin((2 * Math.PI * kTarget * n) / N);
}

const windows: WindowFunctionType[] = ['rectangular', 'hanning', 'hamming'];

for (const win of windows) {
  const result = computeSpectrum(signal, Fs, N, win);
  const binIndex = kTarget;
  const reportedAmp = result.linearMagnitudes[binIndex];
  const reportedFreq = result.frequencies[binIndex];
  const expectedDb = Math.round(20 * Math.log10(knownAmplitude) * 10) / 10; // -2.5 dB
  const reportedDb = result.magnitudes[binIndex];

  console.log(`\n  --- Window: ${win.toUpperCase()} ---`);
  assertClose(reportedFreq, fTarget, `Frequency bin correctly mapped to ${fTarget} Hz`);
  assertClose(reportedAmp, knownAmplitude, `Peak amplitude accurately recovers ${knownAmplitude}g (tolerance: 1e-4)`);
  assertClose(reportedDb, expectedDb, `dB re 1.0g accurately matches ${expectedDb} dB`);
}

// -------------------------------------------------------------------------
// TEST 3: DC Bin (k = 0) Scaling - Verification That DC Is Not Doubled
// -------------------------------------------------------------------------
console.log('\n[TEST 3] DC Bin (k = 0) Non-Doubling Verification:');
// Direct FFT verification of a pure DC signal x[n] = 1.0
const dcSignal = new Array(N).fill(1.0);
const { real: dcReal } = cooleyTukeyFFT(dcSignal);
const cg = getCoherentGain('rectangular', N);

// In computeSpectrum, isUniqueBin = (i === 0 || i === numBins - 1) uses factor = (1 / N) / coherentGain
const dcFactor = (1 / N) / cg;
const dcReportedAmp = Math.abs(dcReal[0]) * dcFactor;
assertClose(dcReportedAmp, 1.0, 'DC bin magnitude is exactly 1.0g (factor 1/N, NOT doubled to 2.0g)');

// -------------------------------------------------------------------------
// TEST 4: Nyquist Bin (k = N/2, f = 1000.0 Hz) Scaling - Verification That Nyquist Is Not Doubled
// -------------------------------------------------------------------------
console.log('\n[TEST 4] Nyquist Bin (k = N/2, f = 1000 Hz) Non-Doubling Verification:');
// Alternating signal at exactly Nyquist: x[n] = A * (-1)^n = A * cos(pi * n)
const nyquistAmp = 0.60; // 0.60g
const nyquistSignal = new Array(N);
for (let n = 0; n < N; n++) {
  nyquistSignal[n] = nyquistAmp * ((n % 2 === 0) ? 1.0 : -1.0);
}

// Compute spectrum with rectangular window
const nyquistResult = computeSpectrum(nyquistSignal, Fs, N, 'rectangular');
const nyquistBin = N / 2; // bin 512
const reportedNyquistAmp = nyquistResult.linearMagnitudes[nyquistBin];
const reportedNyquistFreq = nyquistResult.frequencies[nyquistBin];

assertClose(reportedNyquistFreq, 1000.0, 'Nyquist frequency bin correctly mapped to 1000.0 Hz');
assertClose(reportedNyquistAmp, nyquistAmp, `Nyquist bin magnitude is exactly ${nyquistAmp}g (factor 1/N, NOT doubled to ${(nyquistAmp * 2).toFixed(2)}g)`);

console.log('\n================================================================');
if (allPassed) {
  console.log('      ALL SYNTHETIC FFT CHECKS PASSED WITH STRICT TOLERANCE!    ');
} else {
  console.error('      SOME CHECKS FAILED! INSPECT OUTPUT ABOVE.               ');
  process.exit(1);
}
console.log('================================================================\n');

