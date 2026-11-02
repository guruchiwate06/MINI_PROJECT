# Evaluation report

Observations: 828; positives: 59; ambiguous binary labels omitted: 121.

| Metric | Value |
| --- | --- |
| average_precision | 0.3847753358407878 |
| roc_auc | 0.8202155561922814 |
| precision | 0.4594594594594595 |
| recall | 0.288135593220339 |
| f1 | 0.3541666666666667 |
| brier | 0.1377243666510797 |
| negative_log_likelihood | 0.43601685409214536 |
| ece | 0.20918046282573874 |
| aurc_classification_error | 0.062618705460271 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 5,
  "false_positive_events": 15,
  "false_negative_events": 4,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5835945692823914,
  "event_recall": 0.5555555555555556,
  "event_precision": 0.25,
  "false_alarms_per_km": 9.472121394554463,
  "localization_mae_m": 0.9921140704553636,
  "localization_p95_m": 2.8564114163091285,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
