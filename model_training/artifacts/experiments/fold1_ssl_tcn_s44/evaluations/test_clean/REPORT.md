# Evaluation report

Observations: 567; positives: 57; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.24648194322806538 |
| roc_auc | 0.6055727554179566 |
| precision | 0.12406015037593984 |
| recall | 0.5789473684210527 |
| f1 | 0.2043343653250774 |
| brier | 0.09404062566782377 |
| negative_log_likelihood | 0.40053244831953266 |
| ece | 0.1006380552395059 |
| aurc_classification_error | 0.15944098256369907 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 3,
  "false_positive_events": 12,
  "false_negative_events": 6,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.1428445607736528,
  "event_recall": 0.3333333333333333,
  "event_precision": 0.2,
  "false_alarms_per_km": 10.500115599164733,
  "localization_mae_m": 5.14289464347696,
  "localization_p95_m": 8.992052729517908,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
