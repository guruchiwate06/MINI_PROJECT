# Evaluation report

Observations: 567; positives: 57; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.39728253368973426 |
| roc_auc | 0.7632610939112487 |
| precision | 0.6216216216216216 |
| recall | 0.40350877192982454 |
| f1 | 0.48936170212765956 |
| brier | 0.42396603479629236 |
| negative_log_likelihood | 5.118212867515053 |
| ece | 0.42871390642816315 |
| aurc_classification_error | 0.05976322974383733 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 2,
  "false_positive_events": 10,
  "false_negative_events": 7,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.1428445607736528,
  "event_recall": 0.2222222222222222,
  "event_precision": 0.16666666666666666,
  "false_alarms_per_km": 8.750096332637277,
  "localization_mae_m": 3.5814450868730603,
  "localization_p95_m": 3.6900905954859495,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
