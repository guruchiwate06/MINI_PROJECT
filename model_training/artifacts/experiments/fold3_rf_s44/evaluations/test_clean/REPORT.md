# Evaluation report

Observations: 828; positives: 59; ambiguous binary labels omitted: 121.

| Metric | Value |
| --- | --- |
| average_precision | 0.22506361689937585 |
| roc_auc | 0.8074761411474289 |
| precision | 0.2647058823529412 |
| recall | 0.15254237288135594 |
| f1 | 0.1935483870967742 |
| brier | 0.06932947919904317 |
| negative_log_likelihood | 0.2326940133167674 |
| ece | 0.04572091047865425 |
| aurc_classification_error | 0.021379113190150796 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 3,
  "false_positive_events": 16,
  "false_negative_events": 6,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5835945692823914,
  "event_recall": 0.3333333333333333,
  "event_precision": 0.15789473684210525,
  "false_alarms_per_km": 10.103596154191427,
  "localization_mae_m": 5.221317352779579,
  "localization_p95_m": 7.224921917309416,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
