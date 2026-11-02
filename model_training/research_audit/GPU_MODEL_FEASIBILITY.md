Audit date: 2 November 2026 (Asia/Kolkata). Evidence is local unless linked to a primary external source. VERIFIED means the described check passed; it does not imply clinical, safety, or deployment validation.

# Model and compute feasibility

## Available compute and executed test

The user confirms an **RTX 5070 is available for training**; this laptop is the local audit host. Remote 5070 VRAM, drivers and environment were not accessed, so no exact device-memory claim is made. The local machine has an RTX 3050 Laptop 6 GB, approximately 16.87 GB physical RAM, roughly 50 GB free on C: and 1.18 TB on D: at inspection. Its torch is `2.10.0+cpu`, torchvision `0.25.0+cpu`, and CUDA is unavailable to that installation. This is a local environment detail, **not a research blocker**. No package replacement was performed.

A real two-window batch from AtoB_20_1 passed the existing raw CNN on CPU: input `[2,6,1000]`, finite output `[2,2]`, 18,770 parameters. The model's fit wrapper currently keeps data/model on CPU; add explicit device selection and stream batches for remote training. No accuracy was inferred from this forward pass. No pretrained DINOv2/ConvNeXt weights or project model checkpoints were found in the inspected cache/output paths; two unrelated text-model caches exist. Transformers/torchvision are installed, timm is not. No DINO forward pass or complete fusion architecture was executed, so end-to-end DINO+CNN feasibility remains conditional on weight acquisition and a 5070 smoke test.

## Transparent memory estimates, not measurements

| Representation/model | Estimate and assumption |
| --- | --- |
| Raw 0.5 s window | 6 × 1,000 × float32 = 24,000 bytes; batch 64 input = 1.536 MB |
| All cached waveform windows materialized | 18,775 × 24,000 = 450.6 MB, with duplicate overlapping content; prefer lazy reads/memmaps |
| Existing feature matrix | 18,775 × 66 × float32 ≈4.96 MB |
| Frozen 384-D image embeddings | 105,901 × 384 × float32 ≈162.66 MB, or ≈81.33 MB fp16; candidate subsampling reduces this |
| Decoded resized images cached as float32 | 105,901 × 224 × 224 × 3 × 4 ≈63.77 GB; avoid full RAM materialization |
| CNN input batch | Input memory is tiny; activations/optimizer exceed it. Existing model likely needs well under 1 GB for modest batches, but this is an estimate to profile |
| Candidate-token attention | With B=32, 4 heads, 15 vibration queries and 9 image keys, attention-score tensor has 17,280 values (~69 KB float32), excluding projections/gradients |
| Full raw temporal attention | For B=32, 4 heads, length 1,000, naive scores alone are 128 million values (~512 MB); multiple layers/backprop add substantial memory. Patch/downsample first |
| Spectrogram example | Six channels ×129 bins ×12 frames ×float32 ≈37 KB/window for n_fft=256, hop=64, no boundary padding on 1,000 samples |

For any actual model, include weights, gradient, optimizer state, activations and decoder batches in profiling. A rough Adam parameter budget in fp32 training is 16 bytes per trainable parameter before activations (weight, gradient, first and second moments); mixed-precision implementation/master weights can change it. Frozen feature extraction removes optimizer and gradient state for that backbone. Do not equate input tensor size with peak VRAM or claim these calculations are measured 5070 capacity.

On the training machine, run `nvidia-smi` and capture torch/CUDA versions, verify one real raw batch, then a frozen pretrained image batch at 224 px. Record peak allocated and reserved memory, throughput, finite gradients and logits. Start with batches 8–16 for image extraction and 32–64 for tiny raw/embedding fusion; adjust from actual measurements. Cache features and pin the encoder revision. These are starting configurations, not tested hardware limits.

## Architecture assessment

| Method | Feasibility | Evidence and caveat |
| --- | --- | --- |
| Raw CNN / small TCN | VERIFIED FEASIBLE for raw CNN execution; TCN conditional | Six synchronized finite channels and CPU smoke test; no validated deep generalization score yet |
| Frozen DINOv2 ViT-S/14 / ConvNeXt | CONDITIONALLY FEASIBLE | Actual images decode; pretrained weights and 5070 forward pass still required; fine-tuning is not justified initially |
| Cross-attention | CONDITIONALLY FEASIBLE | Computationally small with candidate tokens; requires real pair calibration and strict candidate-support boundaries |
| GCN/GATv2/ST-GNN | CONDITIONALLY FEASIBLE computationally | Short corridor supports a line graph only after canonical mapping; branch connectivity and added scientific value not established |
| DANN/GRL | CONDITIONALLY FEASIBLE | Same-direction multiple target speeds exist; very few runs per cell, site/type/date confounds, target-data access rules must be explicit |
| Evidential/Dirichlet fusion | CONDITIONALLY FEASIBLE | Easy head implementation does not establish calibrated epistemic uncertainty; nine sites limit calibration/OOD claims |
| Raw unpatched Transformer | Not recommended | Long-token cost and low site diversity; no evidence of advantage over tiny CNN/TCN |

Use normal training statistics only, no normalization over test runs or all runs. Preserve per-channel scale metadata and compare waveform normalization choices. Spectrograms offer an alternate representation but are not a second physical modality. Do not claim new sensors from reshaping one signal into an image. DR-Train pretraining is optional and currently less urgent than split/alignment work.
