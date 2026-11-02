# Decision on the proposed research upgrade

Assessment date: 2 November 2026. This supplements the completed audit; it does not replace its results or mark proposed experiments as completed. The RTX 5070 is the intended training machine.

**Decision: pursue the stronger research direction, with a narrow primary hypothesis.** Implementation is credible. A large improvement on unseen physical defects remains an empirical question. The strongest opportunity is to combine explicitly uncertain spatial correspondence with repeated-pass evidence, and test whether that improves detection efficiency over strong modern single-modality and simple-fusion baselines.

## What the current evidence supports

The existing combined run/site RF reference has average precision 0.3360, F1 0.2815 and ROC-AUC 0.9012. Its confusion matrix contains 19 true-positive, 35 false-positive and 62 false-negative **windows**; these do not determine event recall. This is useful headroom to investigate, not evidence that a deep model will close it. The reference comes from one already inspected split, and must be rerun on any changed cohort.

There are nine physical anomaly sites, 20 vibration runs and 19 runs with nominal same-run image pairing. Physical image alignment remains unvalidated. One image run is wrong-date/duplicated in checked evidence. More frames, augmentation, extra training epochs and pretrained models do not increase the number of independent railway defect sites.

## Three hypotheses, rather than seven equal research contributions

| Priority | Hypothesis | Experiment capable of supporting it |
| --- | --- | --- |
| Primary | Accounting for positional uncertainty improves fusion at unseen sites | Compare late fusion, spatial candidate pooling and uncertainty-aware fusion on identical guarded site/run folds; independently validate correspondence residuals |
| Primary operational evaluation | Repeated observations improve detection with fewer passes at a fixed false-alarm allowance | Freeze the trained detector; pool only held-out inference runs; compare mean, median, maximum and proposed aggregation with equal run counts and conditions |
| Secondary | Self-supervised vibration learning helps when independent labelled defects are scarce | Compare supervised and pretrained versions of the same encoder under site-level label budgets, identical permitted pretraining data and equal selection budgets |

Blur/noise/missing-modality tests are robustness evaluations. Sensor subsets are ablations. GNN, DANN and evidential learning remain optional. A paper does not become stronger merely by including more of these families.

## Necessary corrections to the pasted proposal

1. **Few-label learning:** the existing spatial protocol has only five training anomaly sites. Sampling 10% of overlapping windows from all five still exposes all five defects. Use budgets such as one, two, four and five labelled training sites, with repeated prespecified subsets, and separately report annotation windows, site identities, normal-data availability and validation-label costs. If window-label budgets are also studied, call them a separate task. Unlabelled pretraining cannot include held-out runs or held-out spatial sectors in an inductive claim. “Normal-only” curation itself may consume annotation or location information and must be accounted for.
2. **Repeated passes:** the old combined test has four vibration runs but only three with accepted image pairing. Its multimodal experiment cannot support eight independent passes. Create a separate run allocation for an eight-pass curve, recount conditions/labels and retain sufficient training/validation runs; do not reuse training runs as additional test passes. Adding repeated frame windows from one run does not increase pass count. A frozen model may aggregate new observations from test sites at inference; those observations must not update its weights or training memory bank in the inductive setting.
3. **Site-level AP:** all nine documented anomaly sites are positive. AP over those nine sites alone is not a useful binary ranking evaluation. Use a prespecified population of track segments that includes normal control segments, alongside per-defect event recall, false alarms per kilometre and localization error. Freeze spatial support, aggregation lengths and denominators. Report uncertain/buffer regions explicitly rather than silently deleting difficult alerts from operational metrics.
4. **Cross-run contrastive learning:** matching the same training chainage across runs can teach location identity and background appearance. Omitting GPS from network inputs does not prevent that shortcut. Keep correspondence construction training-only, test unseen sectors, and compare augmentation/time–frequency pretraining with and without same-position positives. Cross-run consistency should not force operating-condition or diagnostic changes to vanish.
5. **Alignment labels:** nominal timestamp offsets are not measured true displacement. Synthetic zero means the recorded alignment, not perfect alignment. Calibrate with independently established landmarks and validate on withheld landmarks; test-anomaly coordinates must not correct their own input positions. Restrict model/candidate support at spatial boundaries, including uncertainty and camera field of view.
6. **Fair budgets:** compare all methods on identical accepted observations, labelled site budgets, training-data access, run counts and false-alarm operating points. Report full-cohort missing-modality fallback separately. The old 1,178-window combined vibration test is not directly comparable with its 846-window nominal paired subset.

## Recommended model development

Use a frozen pretrained vision encoder with patch features and a small multi-scale vibration CNN/TCN. Evaluate DINOv2 and DINOv3 on inner validation only, rather than assuming the newer backbone wins. Select one strong patch-anomaly method, such as AnomalyDINO or FoundAD, as a vision comparator. A supervised vision head is a separate comparator because its supervision differs from normal-reference anomaly detection. Restrict reference memory to permitted training data and remove cone shortcuts.

First establish a raw vibration encoder, then optionally initialize the same encoder with time–frequency self-supervision. The 2026 supporting paper studies machinery/bearing diagnosis; transfer to railway responses is an experiment, not an established result.

For the proposed method, construct a bounded bag of image candidates around each vibration observation. Marginalize or pool over plausible offsets using independently estimated positional uncertainty, rather than committing to one nominal frame. Initially compare fixed distance weighting with attention using the same candidates. At inference, combine repeated-pass scores using a robust calibrated aggregator; compare directly with simple mean/median/max and matched observation budgets. Repeated runs are correlated, so multiplying probabilities as independent evidence can produce overconfidence. Add a learned quality gate only if it improves validation robustness over these controls.

This is a candidate method design, not a claim that uncertainty pooling, attention or repeated measurements are globally novel. The paper's contribution must be the demonstrated interaction under a carefully specified railway setting and a clear distinction from prior alignment/fusion work.

## What would count as going substantially beyond the benchmarks

- Improved segment AP and unseen-defect event recall over the **strongest modern comparator**, rather than only the old RF or the publication's noncomparable 0.73 image AUC.
- More defects detected with the same or fewer held-out passes at the same validation-fixed false-alarm allowance.
- Less degradation under independently measured alignment residuals and declared synthetic shifts.
- Effects that recur across outer site/run folds, with per-site results and uncertainty acknowledging the small number of sites.
- Ablations that distinguish spatial uncertainty, pretraining, repeated-pass aggregation and reliability gating. An improvement caused by more passes or more labelled sites is not an architectural gain.

No numerical success probability or F1 forecast is supported yet. A useful design target should be chosen from the practical alarm/pass budget and then frozen, not picked after seeing the test scores. Even a strong result on these nine sites remains a within-corridor result; additional independent railway sites would materially strengthen claims of broad generalization.

## Decisive pilot and continuation gates

1. Integrate existing source-interval guards and versioned manifests; freeze development and outer evaluation roles; reject the wrong-run images. This uses the completed audit rather than repeating benchmark reconciliation.
2. Measure physical alignment and inspect blinded rail ROI/cone-suppressed samples. If visual defect evidence or correspondence is inadequate, revise the multimodal target before training a complex fusion network.
3. Train the raw vibration baseline on RTX 5070; extract frozen patch features and establish a modern vision baseline. Compare on the same permitted cohort.
4. Run late fusion versus bounded spatial pooling. If neither adds reliable evidence over vibration, investigate visual visibility/alignment before escalating capacity.
5. Add the proposed uncertainty mechanism and held-out repeated-pass aggregation. Require gains against stronger simple counterparts, then test self-supervised initialization under independent-site label budgets.

Use inner groups for these development decisions. Fix the final hypothesis, ablations, thresholds and model-selection rules before outer evaluation. Previously viewed folds remain exploratory and must be disclosed. Neither repeatedly inspecting an outer fold nor changing a random seed produces a pristine test set.

Recommended research question: **Can alignment uncertainty and repeated-pass evidence improve unseen-site railway anomaly detection at a fixed inspection and false-alarm budget?**

Working title: **Railway Anomaly Detection at Unseen Sites through Alignment-Uncertain Multimodal and Repeated-Pass Fusion.** Retain the method wording only if the corresponding experiments support it.

## Primary sources checked for the newer-method claims

- [DINOv3, Meta official release](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/) and [official implementation](https://github.com/facebookresearch/dinov3).
- [AnomalyDINO](https://arxiv.org/abs/2405.14529): first submitted in 2024, revised in 2025; these dates should not be conflated.
- [FoundAD, ICLR 2026 official proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9e47a0bc530cc88b09b7670d2c130a29-Abstract-Conference.html).
- [RadioCore, CVPR Workshops 2026](https://openaccess.thecvf.com/content/CVPR2026W/VISION26/papers/Ali_RadioCore_Few-Shot_Industrial_Anomaly_Segmentation_with_Multi-Scale_Radio_ViT_Features_CVPRW_2026_paper.pdf): a candidate industrial-vision comparator, not a demonstrated railway result.
- [Time–frequency self-supervised contrastive fault diagnosis, IEEE Sensors Journal 2026](https://doi.org/10.1109/JSEN.2026.3715802).
- [Industrial time-series foundation-model applicability study, IEEE Access 2026](https://doi.org/10.1109/ACCESS.2026.3690553): evidence against assuming a generic foundation model automatically wins on mechanical signals.
- [Rail image/axle-box alignment, Zhang et al. 2026](https://link.springer.com/article/10.1007/s10044-026-01620-3) and [visual/inertial fusion and misalignment, Radosavljevic et al. 2026](https://satie.ens-paris-saclay.fr/fr/publications/multimodal-railway-defect-detection-visual-and-inertial-data-early-fusion-strategies): direct prior art that limits broad novelty claims.

The pasted Cycle-CFM and other vaguely specified references were not all independently reviewed in full. They are not needed to justify the selected pilot. None of the sources establishes the proposed performance on Rail-VIVID.
