# Evaluation report

Observations: 687; positives: 56; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.26797597872683454 |
| roc_auc | 0.8141838351822503 |
| precision | 0.10453648915187377 |
| recall | 0.9464285714285714 |
| f1 | 0.1882770870337478 |
| brier | 0.5681712690952386 |
| negative_log_likelihood | 1.566116386254935 |
| ece | 0.6863029262803242 |
| aurc_classification_error | 0.7436191063204334 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 4,
  "false_positive_events": 41,
  "false_negative_events": 5,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5510805166310173,
  "event_recall": 0.4444444444444444,
  "event_precision": 0.08888888888888889,
  "false_alarms_per_km": 26.433186130822495,
  "localization_mae_m": 4.76196904629461,
  "localization_p95_m": 8.839041427427786,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
