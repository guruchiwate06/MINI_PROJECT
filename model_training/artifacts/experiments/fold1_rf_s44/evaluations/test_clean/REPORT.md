# Evaluation report

Observations: 567; positives: 57; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.515548441389734 |
| roc_auc | 0.9081183350533195 |
| precision | 0.3058823529411765 |
| recall | 0.9122807017543859 |
| f1 | 0.4581497797356828 |
| brier | 0.08006079954307174 |
| negative_log_likelihood | 0.2858019217036827 |
| ece | 0.10523382928443346 |
| aurc_classification_error | 0.04372640506167513 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 8,
  "false_positive_events": 26,
  "false_negative_events": 1,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.1428445607736528,
  "event_recall": 0.8888888888888888,
  "event_precision": 0.23529411764705882,
  "false_alarms_per_km": 22.750250464856922,
  "localization_mae_m": 5.263686759509387,
  "localization_p95_m": 8.69002192195482,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
