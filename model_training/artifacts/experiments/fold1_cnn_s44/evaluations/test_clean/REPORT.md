# Evaluation report

Observations: 567; positives: 57; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.4984824630060492 |
| roc_auc | 0.8608875128998967 |
| precision | 0.6428571428571429 |
| recall | 0.3157894736842105 |
| f1 | 0.4235294117647059 |
| brier | 0.07730670700476874 |
| negative_log_likelihood | 0.34656592753314475 |
| ece | 0.07087638937627261 |
| aurc_classification_error | 0.02294146525547012 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 5,
  "false_positive_events": 4,
  "false_negative_events": 4,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.1428445607736528,
  "event_recall": 0.5555555555555556,
  "event_precision": 0.5555555555555556,
  "false_alarms_per_km": 3.5000385330549113,
  "localization_mae_m": 3.407406699085203,
  "localization_p95_m": 6.320667048915379,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
