# Evaluation report

Observations: 687; positives: 56; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.4710475308572106 |
| roc_auc | 0.859038940457324 |
| precision | 0.3291139240506329 |
| recall | 0.4642857142857143 |
| f1 | 0.3851851851851852 |
| brier | 0.08298698729619561 |
| negative_log_likelihood | 0.27417536602009845 |
| ece | 0.11616166052878128 |
| aurc_classification_error | 0.022587603939584564 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 7,
  "false_positive_events": 21,
  "false_negative_events": 2,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5510805166310173,
  "event_recall": 0.7777777777777778,
  "event_precision": 0.25,
  "false_alarms_per_km": 13.538948993835913,
  "localization_mae_m": 3.568269154724021,
  "localization_p95_m": 5.927054351062082,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
