# TRACK-OT++ recovery after the GPU pilot failure

The uploaded run completed one epoch, reported validation AP 0.7916 and then stopped on a nonfinite gradient norm. This does not demonstrate test superiority. The failed run remains intact. No recovery model was trained on the development laptop.

## Changes in this recovery

- Transport matrices are solved as a batch instead of separately per bag, with checked early convergence under the existing iteration cap. Padded and completely unsupported rows/columns avoid differentiating an all-negative-infinity reduction. Tests compare plans and gradients against an independent converged scalar reference.
- The cache reader selects the requested patch tokens before materializing the image-patch array. The old reader loaded all 256 patches before selecting 16. Selected values are unchanged.
- The equivariance term reuses waveform and image projections. It recomputes only image context and the warp, with controlled dropout. It no longer recomputes TCN/FFT/transport/heads twice. This changes the stochastic regularizer's execution and requires a new experiment identity.
- Cycle and contrastive matrix products explicitly run outside autocast. Casting their inputs to float32 alone did not prevent autocast from executing matrix multiplication in FP16.
- Near-empty transport rows remain unmatched instead of amplifying their normalization gradients. Feature normalization has an explicit finite floor.
- Gradient norms use an FP64 sum of squares, which avoids norm overflow when individual FP32 gradients are finite. Actual NaN/Inf parameter gradients still stop the run before an optimizer update. Failures now record parameter names, component losses and source-window IDs.
- A new five-epoch FP32 recovery pilot avoids FP16 during diagnosis. General AMP configuration explicitly selects BF16 by default and uses no loss scaler for BF16. FP16 remains an explicit diagnostic option with a recorded initial scale. Hardware support is checked; there is no silent precision fallback.
- Per-epoch validation predictions are saved. Training and validation CUDA peaks are recorded separately; the previous training-peak field was read after validation reset the CUDA counter.
- `auto_train.py` now requires explicit training-device authorization, runs a finite-gradient preflight, and launches training only. It never evaluates the held-out test set automatically.
- Dashboard TRACK-OT fields now distinguish validation AP from test AP and derive failure status from logs. Hardcoded Brier, speed, latency, checkpoint-size and deployment claims were removed from the affected presentation.

The old log does not identify the exact offending parameter or operation, and its checkpoint/batch payloads are not in Git. These are verified code/numerical-contract repairs, **not proof that the specific RTX failure has been reproduced or eliminated**. Real checkpoint diagnostics on the training device are still required.

## On the training computer

Use a separate checkout if another process is active. Keep `artifacts/track_ot/track_ot_pp_fold1_full_s42` and its original logs/checkpoints. Do not replace its config or use `--resume` across this code change.

From `model_training`, run bounded diagnostics against that computer's local checkpoint:

```powershell
python -m track_ot diagnose --config configs/track_ot_recovery_fp32.json --device cuda --from-run artifacts/track_ot/track_ot_pp_fold1_full_s42 --batches 4 --precisions float32,bfloat16,float16 --component-gradients --anomaly --profile --output reports/recovery_diagnostics_01
```

This command performs forward/backward checks, not training: it never constructs an optimizer, verifies unchanged model weights and never accesses the test partition. It records finite gradients per loss term, parameter names, loader/transfer/forward/auxiliary/backward duration, solver iterations and CUDA allocation. Chrome traces and operator tables identify the measured bottleneck. Any requested unsupported/failed precision returns a nonzero status; inspect the per-precision report rather than treating that as a reason to disable checks.

If the old checkpoint is unavailable, omit `--from-run` to check randomly initialized weights. That cannot reproduce a late-training failure. An old log file alone is not a checkpoint.

After FP32 checks pass, start a NEW, limited recovery experiment:

```powershell
python -m track_ot train --config configs/track_ot_recovery_fp32.json --on-training-device --warm-start artifacts/track_ot/track_ot_pp_fold1_full_s42
```

Warm-start validates data, split, sequence-bag and normalization identity, loads model weights only and resets the optimizer/scheduler/scaler. Its parent checkpoint hash and epoch are saved to `warm_start.json`. It is not an exact resume or an independent seed replication. Omit `--warm-start` for a fresh pilot.

Alternatively, the guarded launcher performs a four-batch FP32 preflight before that same limited training stage:

```powershell
python scripts/auto_train.py --config configs/track_ot_recovery_fp32.json --on-training-device --warm-start artifacts/track_ot/track_ot_pp_fold1_full_s42
```

The five-epoch pilot has patience three. It is a stability/performance gate before any 99-job campaign. Choose a new experiment name for a later longer run. Keep validation model selection separate from the final test evaluation.

## Interpretation and remaining evidence

No GPU speedup factor is claimed from the CPU checks. Measure actual bags/sec and the new timing/trace files. GPU memory allocation does not measure utilization. The PyTorch AMP documentation explains the FP16 gradient range and why floating inputs can still be autocast during matrix multiplication; the profiler provides operator/trace evidence. [AMP](https://docs.pytorch.org/docs/stable/amp.html), [profiler](https://docs.pytorch.org/docs/stable/profiler.html).

The same Rail-VIVID data, original anchor labels and guarded partitions are retained. Marker masking, independently checked alignment, channel-placement ablations and frozen test results still need actual experimental evidence. A stable training run is necessary but does not by itself establish novelty or publication quality.
