# Evaluation report

Observations: 567; positives: 57; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.5535092193302527 |
| roc_auc | 0.855796353629171 |
| precision | 0.1895910780669145 |
| recall | 0.8947368421052632 |
| f1 | 0.3128834355828221 |
| brier | 0.4539077903753401 |
| negative_log_likelihood | 1.826990577394436 |
| ece | 0.5258122217321776 |
| aurc_classification_error | 0.3639493897526147 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 4,
  "false_positive_events": 13,
  "false_negative_events": 5,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.1428445607736528,
  "event_recall": 0.4444444444444444,
  "event_precision": 0.23529411764705882,
  "false_alarms_per_km": 11.375125232428461,
  "localization_mae_m": 3.426761992274571,
  "localization_p95_m": 6.704339810406783,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
