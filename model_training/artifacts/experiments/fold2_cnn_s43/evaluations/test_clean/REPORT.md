# Evaluation report

Observations: 687; positives: 56; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.23337183663846833 |
| roc_auc | 0.7945155082635272 |
| precision | 0.09228187919463088 |
| recall | 0.9821428571428571 |
| f1 | 0.1687116564417178 |
| brier | 0.5759946662746152 |
| negative_log_likelihood | 1.8193219043012496 |
| ece | 0.6826385424182357 |
| aurc_classification_error | 0.7841984850565975 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 3,
  "false_positive_events": 23,
  "false_negative_events": 6,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5510805166310173,
  "event_recall": 0.3333333333333333,
  "event_precision": 0.11538461538461539,
  "false_alarms_per_km": 14.828372707534571,
  "localization_mae_m": 3.8313025025219076,
  "localization_p95_m": 7.872323519313874,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
