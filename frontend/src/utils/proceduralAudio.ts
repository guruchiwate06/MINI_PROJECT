/**
 * PROCEDURAL AUDIO SYNTHESIS ENGINE
 * 
 * SCIENTIFIC INTEGRITY NOTICE:
 * This audio engine uses the Web Audio API to procedurally synthesize 
 * acoustic telemetry derived from accelerometer vibration amplitudes and vehicle speed.
 * It is EXCLUSIVELY labeled as PROCEDURAL AUDIO and is NEVER presented as original
 * acoustic microphone recordings from the Orbisonia testbed.
 */

class ProceduralAudioEngine {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private rumbleOsc: OscillatorNode | null = null;
  private rumbleGain: GainNode | null = null;
  private frictionFilter: BiquadFilterNode | null = null;
  private frictionGain: GainNode | null = null;
  private noiseSource: AudioBufferSourceNode | null = null;
  private impactFilter: BiquadFilterNode | null = null;
  private impactGain: GainNode | null = null;

  private isRunning: boolean = false;
  private isMuted: boolean = false;
  private currentVolume: number = 0.3;

  /**
   * Initialize audio context on explicit user gesture
   */
  public init(): boolean {
    if (this.ctx) return true;
    try {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.ctx = new AudioCtx();

      // Master output node
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.setValueAtTime(this.isMuted ? 0 : this.currentVolume, this.ctx.currentTime);
      this.masterGain.connect(this.ctx.destination);

      // 1. Low-frequency track/bogie rumble oscillator
      this.rumbleOsc = this.ctx.createOscillator();
      this.rumbleOsc.type = 'triangle';
      this.rumbleOsc.frequency.setValueAtTime(32.0, this.ctx.currentTime);

      this.rumbleGain = this.ctx.createGain();
      this.rumbleGain.gain.setValueAtTime(0.2, this.ctx.currentTime);
      this.rumbleOsc.connect(this.rumbleGain);
      this.rumbleGain.connect(this.masterGain);
      this.rumbleOsc.start();

      // 2. Wheel-rail rolling friction (pink noise generator)
      const bufferSize = this.ctx.sampleRate * 2;
      const noiseBuffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
      const output = noiseBuffer.getChannelData(0);
      let b0 = 0, b1 = 0, b2 = 0;
      for (let i = 0; i < bufferSize; i++) {
        const white = Math.random() * 2 - 1;
        b0 = 0.99886 * b0 + white * 0.0555179;
        b1 = 0.99332 * b1 + white * 0.0750759;
        b2 = 0.96900 * b2 + white * 0.1538520;
        output[i] = (b0 + b1 + b2) * 0.15;
      }

      this.noiseSource = this.ctx.createBufferSource();
      this.noiseSource.buffer = noiseBuffer;
      this.noiseSource.loop = true;

      this.frictionFilter = this.ctx.createBiquadFilter();
      this.frictionFilter.type = 'bandpass';
      this.frictionFilter.frequency.setValueAtTime(380.0, this.ctx.currentTime);
      this.frictionFilter.Q.setValueAtTime(1.5, this.ctx.currentTime);

      this.frictionGain = this.ctx.createGain();
      this.frictionGain.gain.setValueAtTime(0.15, this.ctx.currentTime);

      this.noiseSource.connect(this.frictionFilter);
      this.frictionFilter.connect(this.frictionGain);
      this.frictionGain.connect(this.masterGain);
      this.noiseSource.start();

      // 3. Rail Joint Impact Resonator (High-Q shock impact)
      this.impactFilter = this.ctx.createBiquadFilter();
      this.impactFilter.type = 'peaking';
      this.impactFilter.frequency.setValueAtTime(240.0, this.ctx.currentTime);
      this.impactFilter.Q.setValueAtTime(6.0, this.ctx.currentTime);
      this.impactFilter.gain.setValueAtTime(0.0, this.ctx.currentTime);

      this.impactGain = this.ctx.createGain();
      this.impactGain.gain.setValueAtTime(0.0, this.ctx.currentTime);

      this.frictionFilter.connect(this.impactFilter);
      this.impactFilter.connect(this.impactGain);
      this.impactGain.connect(this.masterGain);

      this.isRunning = true;
      return true;
    } catch (e) {
      console.warn('Web Audio API initialization failed:', e);
      return false;
    }
  }

  /**
   * Dynamically update acoustic properties based on active 6-channel telemetry
   */
  public updateTelemetry(speedKmh: number, ch1_g: number, ch2_axle_g: number, isAnomaly: boolean): void {
    if (!this.ctx || !this.isRunning || this.ctx.state === 'suspended') return;

    const now = this.ctx.currentTime;
    const safeTime = now + 0.03;

    // Modulate rumble pitch with vehicle speed (20 km/h ~ 30 Hz, 27.5 km/h ~ 45 Hz)
    if (this.rumbleOsc) {
      const targetFreq = Math.max(20, Math.min(65, 15 + speedKmh * 0.9));
      this.rumbleOsc.frequency.setTargetAtTime(targetFreq, now, 0.05);
    }

    // Modulate friction noise with axle box high-frequency vibration
    if (this.frictionGain && this.frictionFilter) {
      const axleIntensity = Math.min(1.0, Math.abs(ch2_axle_g) * 1.5);
      const targetGain = 0.1 + axleIntensity * 0.25;
      this.frictionGain.gain.setTargetAtTime(targetGain, now, 0.05);

      const targetFilterFreq = 300 + Math.abs(ch1_g) * 350;
      this.frictionFilter.frequency.setTargetAtTime(targetFilterFreq, now, 0.05);
    }

    // Trigger audible transient ping/impact when rail joint shock occurs
    if (this.impactGain && this.impactFilter) {
      if (isAnomaly || Math.abs(ch2_axle_g) > 0.8) {
        // Impact strike
        this.impactGain.gain.setTargetAtTime(0.4, now, 0.01);
        this.impactGain.gain.setTargetAtTime(0.0, safeTime, 0.12);
        this.impactFilter.gain.setTargetAtTime(14.0, now, 0.01);
      } else {
        this.impactGain.gain.setTargetAtTime(0.0, now, 0.05);
      }
    }
  }

  public setVolume(volume: number): void {
    this.currentVolume = Math.max(0, Math.min(1, volume));
    if (this.masterGain && this.ctx && !this.isMuted) {
      this.masterGain.gain.setTargetAtTime(this.currentVolume, this.ctx.currentTime, 0.03);
    }
  }

  public setMuted(muted: boolean): void {
    this.isMuted = muted;
    if (this.masterGain && this.ctx) {
      this.masterGain.gain.setTargetAtTime(muted ? 0 : this.currentVolume, this.ctx.currentTime, 0.03);
    }
  }

  public pause(): void {
    if (this.ctx && this.ctx.state === 'running') {
      this.ctx.suspend();
    }
  }

  public resume(): void {
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  public stop(): void {
    this.pause();
  }

  public destroy(): void {
    if (this.ctx) {
      this.ctx.close();
      this.ctx = null;
      this.isRunning = false;
    }
  }
}

export const proceduralAudio = new ProceduralAudioEngine();
