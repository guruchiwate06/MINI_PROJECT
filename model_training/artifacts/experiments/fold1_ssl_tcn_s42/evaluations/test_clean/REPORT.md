# Evaluation report

Observations: 567; positives: 57; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.31853936803188937 |
| roc_auc | 0.6148950808393533 |
| precision | 0.75 |
| recall | 0.15789473684210525 |
| f1 | 0.2608695652173913 |
| brier | 0.10216403767660509 |
| negative_log_likelihood | 0.692049644597136 |
| ece | 0.09216245220800448 |
| aurc_classification_error | 0.1280118621545398 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 2,
  "false_positive_events": 4,
  "false_negative_events": 7,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.1428445607736528,
  "event_recall": 0.2222222222222222,
  "event_precision": 0.3333333333333333,
  "false_alarms_per_km": 3.5000385330549113,
  "localization_mae_m": 3.826153784566017,
  "localization_p95_m": 6.643562220370115,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
