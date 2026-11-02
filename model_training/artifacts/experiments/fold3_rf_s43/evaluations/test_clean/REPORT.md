# Evaluation report

Observations: 828; positives: 59; ambiguous binary labels omitted: 121.

| Metric | Value |
| --- | --- |
| average_precision | 0.24656715747928074 |
| roc_auc | 0.8299354213043574 |
| precision | 0.2962962962962963 |
| recall | 0.13559322033898305 |
| f1 | 0.18604651162790697 |
| brier | 0.06611343491335918 |
| negative_log_likelihood | 0.22077144937624096 |
| ece | 0.03417785952196471 |
| aurc_classification_error | 0.018557550438430232 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 3,
  "false_positive_events": 13,
  "false_negative_events": 6,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5835945692823914,
  "event_recall": 0.3333333333333333,
  "event_precision": 0.1875,
  "false_alarms_per_km": 8.209171875280536,
  "localization_mae_m": 5.221317352779579,
  "localization_p95_m": 7.224921917309416,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
