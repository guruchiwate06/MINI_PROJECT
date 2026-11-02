# Evaluation report

Observations: 828; positives: 59; ambiguous binary labels omitted: 121.

| Metric | Value |
| --- | --- |
| average_precision | 0.21911848989832447 |
| roc_auc | 0.8014150007714179 |
| precision | 0.2903225806451613 |
| recall | 0.15254237288135594 |
| f1 | 0.2 |
| brier | 0.06818449082217092 |
| negative_log_likelihood | 0.230568934153263 |
| ece | 0.031647243752826903 |
| aurc_classification_error | 0.021545829066630974 |

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
