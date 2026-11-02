# Evaluation report

Observations: 687; positives: 56; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.463548836043421 |
| roc_auc | 0.8630009055920308 |
| precision | 0.0 |
| recall | 0.0 |
| f1 | 0.0 |
| brier | 0.08202110508813856 |
| negative_log_likelihood | 0.2699917050275449 |
| ece | 0.11748495480158225 |
| aurc_classification_error | 0.02203475816015052 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 0,
  "false_positive_events": 0,
  "false_negative_events": 9,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5510805166310173,
  "event_recall": 0.0,
  "event_precision": 0.0,
  "false_alarms_per_km": 0.0,
  "localization_mae_m": null,
  "localization_p95_m": null,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
