# Evaluation report

Observations: 567; positives: 57; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.343744930654167 |
| roc_auc | 0.7396628826969385 |
| precision | 0.18705035971223022 |
| recall | 0.45614035087719296 |
| f1 | 0.2653061224489796 |
| brier | 0.0831046410095848 |
| negative_log_likelihood | 0.42713602760331365 |
| ece | 0.06481394917704787 |
| aurc_classification_error | 0.06434643521695646 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 5,
  "false_positive_events": 25,
  "false_negative_events": 4,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.1428445607736528,
  "event_recall": 0.5555555555555556,
  "event_precision": 0.16666666666666666,
  "false_alarms_per_km": 21.875240831593196,
  "localization_mae_m": 1.5988877793123435,
  "localization_p95_m": 2.678349074187372,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
