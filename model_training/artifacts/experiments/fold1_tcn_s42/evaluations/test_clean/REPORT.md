# Evaluation report

Observations: 567; positives: 57; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.3387635745459321 |
| roc_auc | 0.7007567939456485 |
| precision | 0.5925925925925926 |
| recall | 0.2807017543859649 |
| f1 | 0.38095238095238093 |
| brier | 0.0813231352508372 |
| negative_log_likelihood | 0.3678799593847897 |
| ece | 0.06827125918574568 |
| aurc_classification_error | 0.05815381248082645 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 5,
  "false_positive_events": 7,
  "false_negative_events": 4,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.1428445607736528,
  "event_recall": 0.5555555555555556,
  "event_precision": 0.4166666666666667,
  "false_alarms_per_km": 6.125067432846095,
  "localization_mae_m": 3.464889958608461,
  "localization_p95_m": 6.305718545433932,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
