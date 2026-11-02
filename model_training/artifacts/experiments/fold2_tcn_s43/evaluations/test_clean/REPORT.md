# Evaluation report

Observations: 687; positives: 56; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.23932293181722175 |
| roc_auc | 0.6869481548562373 |
| precision | 0.36585365853658536 |
| recall | 0.26785714285714285 |
| f1 | 0.30927835051546393 |
| brier | 0.15169397415111135 |
| negative_log_likelihood | 0.4787764926748915 |
| ece | 0.18424924636944365 |
| aurc_classification_error | 0.08060524164711408 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 3,
  "false_positive_events": 10,
  "false_negative_events": 6,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5510805166310173,
  "event_recall": 0.3333333333333333,
  "event_precision": 0.23076923076923078,
  "false_alarms_per_km": 6.447118568493292,
  "localization_mae_m": 5.047242163572473,
  "localization_p95_m": 8.039756334399124,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
