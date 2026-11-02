# Evaluation report

Observations: 828; positives: 59; ambiguous binary labels omitted: 121.

| Metric | Value |
| --- | --- |
| average_precision | 0.411852434536374 |
| roc_auc | 0.713649688126777 |
| precision | 0.09925558312655088 |
| recall | 0.6779661016949152 |
| f1 | 0.17316017316017315 |
| brier | 0.21865645463095856 |
| negative_log_likelihood | 0.5871810491038345 |
| ece | 0.2763454218148929 |
| aurc_classification_error | 0.14700003298802763 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 5,
  "false_positive_events": 13,
  "false_negative_events": 4,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5835945692823914,
  "event_recall": 0.5555555555555556,
  "event_precision": 0.2777777777777778,
  "false_alarms_per_km": 8.209171875280536,
  "localization_mae_m": 1.652827800957948,
  "localization_p95_m": 4.316090189118494,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
