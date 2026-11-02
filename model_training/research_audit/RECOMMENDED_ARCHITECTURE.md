Audit date: 2 November 2026 (Asia/Kolkata). Evidence is local unless linked to a primary external source. VERIFIED means the described check passed; it does not imply clinical, safety, or deployment validation.

# What to build

**Primary contribution:** a reproducible benchmark of railway anomaly-site detection under physical-location and run shifts, with explicitly measured image-alignment uncertainty. **Primary first deep model:** the existing lightweight six-channel raw CNN. **First multimodal model:** frozen DINOv2 ViT-S/14 features plus that CNN, with a small concatenation classifier and late-fusion control. A true residual/dilated TCN is optional if the existing CNN proves inadequate on validation; do not rename the existing sequential CNN a TCN implementation.

## Six proposed research experiments, ranked

| Rank | Experiment | Recommendation | Scientific value / data fit / main risk / reproducibility / mini-project scope |
| --- | --- | --- | --- |
| 1 | F: strict generalization benchmark | Primary, mandatory | Directly exposes verified evaluation problems; low compute, reusable artifacts, strong reproducibility opportunity; nine sites constrain external claims |
| 2 | E: sensor placement analysis | Essential compact ablation | Six real placements, same labels/splits, low compute; publication already studies channel responses, so classification subset novelty alone is modest |
| 3 | A: deep multimodal fusion | Conditional main model study | Real data in 19 paired runs; first simple fusion, then attention only if alignment controls justify it; highest pair/shortcut risk |
| 4 | D: operating-condition robustness | Secondary controlled experiment | Matched-direction cells exist; evaluate held-out speed with run/site controls before DANN; only four nominal repeats/cell and date confounds |
| 5 | B: uncertainty/reliability | Optional evaluation extension | Missing channels/blur/noise can be stress-tested; temperature scaling/ensemble first; small site count makes calibration claims fragile |
| 6 | C: spatial-temporal GNN | Defer | Short single corridor, no validated branch topology, likely competing with smoothing; more complexity and leakage exposure without proven benefit |

This ranking is qualitative, not a fabricated novelty score. F/E are strongest in data suitability, baseline availability, compute cost and reproducibility. A has the best conditional multimodal contribution but requires more work. D/B/C add selection burden and reduce mini-project feasibility if pursued simultaneously. A scientifically useful system can display relative track distance, anomaly score and uncertainty/missing-modality status without pretending to forecast repairs.

## Model sequence and baselines

1. **RF reference:** preserve existing 66-feature implementation, rerun only on repaired/frozen cohorts. Include GPS-only and GPS+vibration as diagnostic controls, not default production features.
2. **Raw CNN:** `[B,6,1000]`, train-only per-channel normalization, existing 18,770-parameter network, weighted cross-entropy, deterministic data order/seed, validation AP for checkpoint selection and validation-selected operating threshold. Store complete logits and supports. Evaluate 0.5 s versus a small prespecified longer-context choice only in inner validation.
3. **Vision:** freeze pretrained small encoder; rail ROI at documented 224 px transform, normalized to its official preprocessing; 384-D DINO embeddings if using that model. A linear/small MLP head and a normal-only anomaly detector provide supervision-matched controls. Do not train a large vision model from scratch on nine sites.
4. **Simple multimodal:** project raw and frozen image embeddings to modest widths, concatenate, MLP classifier; compare calibrated score-level late fusion and mean/distance-weighted pooling of nearby image candidates. Test identical accepted rows for every modality.
5. **Conditional spatial attention:** retain CNN temporal tokens rather than only its final global average, project to 128 dimensions, attend to a bounded set of corrected/uncertain image candidates with relative-distance encodings and candidate masks. Freeze vision initially. The image candidate list and allowed distance prior must be constructed without held-out anomaly labels. Keep dimensions and image-candidate budget fixed across pooling/attention comparisons. Add an alignment supervision/loss only where independently labelled correspondences exist.

No target F1 is used as an acceptance gate. The current hardest RF F1 0.2815/AP 0.3360 is a reference on its exact cohort, not a lower bound or performance promise. Compare against newly rerun RF on each new cohort. Negative fusion results remain scientifically informative if uncertainty, shortcuts and evaluation composition are controlled.

## Required ablations and reporting

Use vibration-only, vision-only, GPS-only diagnostic, concatenation, late fusion, distance pooling and optional attention; nominal versus independently corrected alignment; cone-present versus suppressed; synthetic offset curves; identical source-support guards; axle 2/4, bogie 1/3, body 5/6, all six; measured/estimated speed conditioning versus none; multiple frozen seeds and site/run folds. The sensor-subset study must keep the observation cohort, hyperparameter budget and split identities fixed. Report inference cost and spectral descriptive evidence without claiming classification proves a physical law.

Primary metrics: average precision, per-site event recall, false alarms per physical kilometre/run and localization error with explicit tolerance. Also report ROC-AUC, precision/recall/F1 at one validation-fixed threshold. Merge repeated adjacent alerts into events with validation-fixed distance/time parameters; match predicted and true events one-to-one. Define “localization” separately for proximity labels and visually adjudicated defect locations. Include counts, prevalence and cluster uncertainty; do not silently remove difficult sites.

For missing-modality reliability, compare signal-only fallback, modality dropout, temperature scaling, a small deep ensemble and optional evidential head. Use NLL/Brier score, reliability diagrams, risk–coverage/AURC and ECE with stated bins, stratified by site/run and corruption type. Synthetic blur/darkening/noise/channel dropout must be labelled synthetic; severity should come from observed brightness/noise distributions or explicit sensitivity ranges. No real failure prevalence or calibrated epistemic interpretation is established by synthetic tests alone.

For any later graph experiment use a verified line graph, no imagined turnout branches. Compare local scores, neighbour concatenation, Gaussian smoothing, distance-axis 1D convolution, and GNN with normal/shuffled/removed edges. Prevent test-label propagation and cross-run summaries from using test traversals in training. For DANN, distinguish domain generalization with no target data from adaptation that uses unlabeled target-train data; never use the final target test set for representation fitting.
