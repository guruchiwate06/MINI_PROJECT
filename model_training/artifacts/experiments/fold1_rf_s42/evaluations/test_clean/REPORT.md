# Evaluation report

Observations: 567; positives: 57; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.5114213326470147 |
| roc_auc | 0.9228070175438596 |
| precision | 0.3216374269005848 |
| recall | 0.9649122807017544 |
| f1 | 0.4824561403508772 |
| brier | 0.07813000282965203 |
| negative_log_likelihood | 0.27906416889288604 |
| ece | 0.10486420707993306 |
| aurc_classification_error | 0.04075192041406448 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 6,
  "false_positive_events": 27,
  "false_negative_events": 3,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.1428445607736528,
  "event_recall": 0.6666666666666666,
  "event_precision": 0.18181818181818182,
  "false_alarms_per_km": 23.62526009812065,
  "localization_mae_m": 4.676598335289498,
  "localization_p95_m": 8.892427920387519,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
