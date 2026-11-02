# Evaluation report

Observations: 567; positives: 57; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.4973978463134968 |
| roc_auc | 0.9043687650498796 |
| precision | 0.242152466367713 |
| recall | 0.9473684210526315 |
| f1 | 0.38571428571428573 |
| brier | 0.07998784966816615 |
| negative_log_likelihood | 0.2842580285155285 |
| ece | 0.10474175016055837 |
| aurc_classification_error | 0.08113507635777288 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 7,
  "false_positive_events": 26,
  "false_negative_events": 2,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.1428445607736528,
  "event_recall": 0.7777777777777778,
  "event_precision": 0.21212121212121213,
  "false_alarms_per_km": 22.750250464856922,
  "localization_mae_m": 4.423276618541848,
  "localization_p95_m": 8.791224921171167,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
