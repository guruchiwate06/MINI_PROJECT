# Evaluation report

Observations: 828; positives: 59; ambiguous binary labels omitted: 121.

| Metric | Value |
| --- | --- |
| average_precision | 0.3460466251131391 |
| roc_auc | 0.8536950915783211 |
| precision | 0.375 |
| recall | 0.15254237288135594 |
| f1 | 0.21686746987951808 |
| brier | 0.05713072196156706 |
| negative_log_likelihood | 0.20378968672190803 |
| ece | 0.03295436173617564 |
| aurc_classification_error | 0.01933552331187506 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 4,
  "false_positive_events": 19,
  "false_negative_events": 5,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5835945692823914,
  "event_recall": 0.4444444444444444,
  "event_precision": 0.17391304347826086,
  "false_alarms_per_km": 11.99802043310232,
  "localization_mae_m": 1.9459268814348434,
  "localization_p95_m": 3.188984256052083,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
