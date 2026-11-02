/**
 * Mathematical Fast Fourier Transform (Cooley-Tukey Radix-2 DIT FFT)
 * and Spectral Feature Extraction for 2,000 Hz Railway Vibration Analysis
 */

export type WindowFunctionType = 'hanning' | 'hamming' | 'rectangular';

export interface SpectralAnalysisResult {
  frequencies: number[];       // Frequency bins in Hz [0, Nyquist]
  magnitudes: number[];        // Magnitude in dB (normalized)
  linearMagnitudes: number[];  // Linear amplitude in g
  dominantFrequency: number;   // Peak frequency in Hz
  peakValue: number;           // Maximum peak magnitude
  spectralCentroid: number;    // Center of mass of the spectrum in Hz
  bandEnergies: {
    low: number;   // 0 - 50 Hz (bogie bounce, pitch, hunting)
    mid: number;   // 50 - 200 Hz (track corrugation, sleeper pass)
    high: number;  // 200 - 500 Hz (axle impact resonance)
    ultra: number; // 500 - 1000 Hz (high-frequency shock acoustic ring)
  };
  topPeaks: Array<{ frequency: number; magnitude: number }>;
}

/**
 * Apply window function to minimize spectral leakage
 */
export function applyWindow(data: number[], windowType: WindowFunctionType): number[] {
  const N = data.length;
  const out = new Float64Array(N);

  for (let i = 0; i < N; i++) {
    let weight = 1.0;
    if (windowType === 'hanning') {
      weight = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1)));
    } else if (windowType === 'hamming') {
      weight = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (N - 1));
    }
    out[i] = data[i] * weight;
  }
  return Array.from(out);
}

/**
 * Radix-2 Cooley-Tukey FFT implementation
 * Input length must be a power of 2
 */
export function cooleyTukeyFFT(real: number[], imag?: number[]): { real: number[]; imag: number[] } {
  const n = real.length;
  if ((n & (n - 1)) !== 0) {
    throw new Error(`FFT length must be a power of 2, received ${n}`);
  }

  const outReal = new Float64Array(real);
  const outImag = imag ? new Float64Array(imag) : new Float64Array(n);

  // Bit-reversal permutation
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      const tempR = outReal[i];
      outReal[i] = outReal[j];
      outReal[j] = tempR;

      const tempI = outImag[i];
      outImag[i] = outImag[j];
      outImag[j] = tempI;
    }
    let k = n >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }

  // Butterfly computation
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wStepR = Math.cos(angle);
    const wStepI = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let wR = 1.0;
      let wI = 0.0;

      for (let k = 0; k < halfLen; k++) {
        const uR = outReal[i + k];
        const uI = outImag[i + k];

        const tR = wR * outReal[i + k + halfLen] - wI * outImag[i + k + halfLen];
        const tI = wR * outImag[i + k + halfLen] + wI * outReal[i + k + halfLen];

        outReal[i + k] = uR + tR;
        outImag[i + k] = uI + tI;
        outReal[i + k + halfLen] = uR - tR;
        outImag[i + k + halfLen] = uI - tI;

        const nextWR = wR * wStepR - wI * wStepI;
        wI = wR * wStepI + wI * wStepR;
        wR = nextWR;
      }
    }
  }

  return { real: Array.from(outReal), imag: Array.from(outImag) };
}

/**
 * Compute real single-sided power spectrum up to Nyquist frequency
 */
export function computeSpectrum(
  samples: number[],
  samplingRateHz: number = 2000,
  windowSize: number = 1024,
  windowType: WindowFunctionType = 'hanning'
): SpectralAnalysisResult {
  // Ensure power of 2 length
  let N = windowSize;
  while ((N & (N - 1)) !== 0) {
    N = 1 << Math.floor(Math.log2(N));
  }

  // Slice or zero-pad input to N
  let windowedInput: number[];
  if (samples.length >= N) {
    // Take central region
    const start = Math.floor((samples.length - N) / 2);
    windowedInput = samples.slice(start, start + N);
  } else {
    windowedInput = [...samples, ...new Array(N - samples.length).fill(0)];
  }

  // Remove DC bias
  const mean = windowedInput.reduce((acc, v) => acc + v, 0) / N;
  const zeroMean = windowedInput.map((v) => v - mean);

  // Windowing
  const windowed = applyWindow(zeroMean, windowType);

  // Compute FFT
  const { real, imag } = cooleyTukeyFFT(windowed);

  // Single-sided spectrum (0 to N/2)
  const numBins = Math.floor(N / 2);
  const frequencies = new Array(numBins);
  const linearMagnitudes = new Array(numBins);
  const magnitudes = new Array(numBins);

  const freqStep = samplingRateHz / N;
  let maxMag = -Infinity;
  let domFreq = 0;
  let weightedFreqSum = 0;
  let totalEnergy = 1e-12;

  let bandLow = 0;   // 0 - 50 Hz
  let bandMid = 0;   // 50 - 200 Hz
  let bandHigh = 0;  // 200 - 500 Hz
  let bandUltra = 0; // 500 - 1000 Hz

  for (let i = 0; i < numBins; i++) {
    const f = i * freqStep;
    frequencies[i] = Math.round(f * 10) / 10;

    // Amplitude scaled by 2/N (except DC bin at i=0)
    const factor = i === 0 ? 1 / N : 2 / N;
    const mag = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]) * factor;
    linearMagnitudes[i] = mag;

    // dB conversion relative to full-scale 1.0g
    const magDb = 20 * Math.log10(Math.max(mag, 1e-6));
    magnitudes[i] = Math.round(magDb * 10) / 10;

    const power = mag * mag;
    totalEnergy += power;
    weightedFreqSum += f * power;

    if (i > 1 && mag > maxMag) {
      maxMag = mag;
      domFreq = f;
    }

    // Band accumulation
    if (f < 50) bandLow += power;
    else if (f < 200) bandMid += power;
    else if (f < 500) bandHigh += power;
    else bandUltra += power;
  }

  // Peak picking (find top local maxima)
  const peaks: Array<{ frequency: number; magnitude: number }> = [];
  for (let i = 2; i < numBins - 2; i++) {
    if (
      linearMagnitudes[i] > linearMagnitudes[i - 1] &&
      linearMagnitudes[i] > linearMagnitudes[i - 2] &&
      linearMagnitudes[i] > linearMagnitudes[i + 1] &&
      linearMagnitudes[i] > linearMagnitudes[i + 2]
    ) {
      peaks.push({
        frequency: frequencies[i],
        magnitude: linearMagnitudes[i]
      });
    }
  }

  peaks.sort((a, b) => b.magnitude - a.magnitude);
  const topPeaks = peaks.slice(0, 5);

  return {
    frequencies,
    magnitudes,
    linearMagnitudes,
    dominantFrequency: Math.round(domFreq * 10) / 10,
    peakValue: Math.round(maxMag * 1000) / 1000,
    spectralCentroid: Math.round((weightedFreqSum / totalEnergy) * 10) / 10,
    bandEnergies: {
      low: Math.round((bandLow / totalEnergy) * 100),
      mid: Math.round((bandMid / totalEnergy) * 100),
      high: Math.round((bandHigh / totalEnergy) * 100),
      ultra: Math.round((bandUltra / totalEnergy) * 100)
    },
    topPeaks
  };
}
