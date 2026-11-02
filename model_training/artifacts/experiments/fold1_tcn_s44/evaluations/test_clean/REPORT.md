# Evaluation report

Observations: 567; positives: 57; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.44427422181863857 |
| roc_auc | 0.664671482628139 |
| precision | 0.5945945945945946 |
| recall | 0.38596491228070173 |
| f1 | 0.46808510638297873 |
| brier | 0.07395543465347328 |
| negative_log_likelihood | 0.769068158418482 |
| ece | 0.07401459779243831 |
| aurc_classification_error | 0.07478413155809814 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 4,
  "false_positive_events": 9,
  "false_negative_events": 5,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.1428445607736528,
  "event_recall": 0.4444444444444444,
  "event_precision": 0.3076923076923077,
  "false_alarms_per_km": 7.87508669937355,
  "localization_mae_m": 2.0079779750357076,
  "localization_p95_m": 3.8054056632387696,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
