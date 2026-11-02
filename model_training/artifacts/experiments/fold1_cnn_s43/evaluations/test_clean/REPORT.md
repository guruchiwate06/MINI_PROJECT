# Evaluation report

Observations: 567; positives: 57; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.504181756979144 |
| roc_auc | 0.8671138630890952 |
| precision | 0.32432432432432434 |
| recall | 0.631578947368421 |
| f1 | 0.42857142857142855 |
| brier | 0.17028288023723487 |
| negative_log_likelihood | 2.6398622146214064 |
| ece | 0.17567933946588862 |
| aurc_classification_error | 0.060663873251156866 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 2,
  "false_positive_events": 18,
  "false_negative_events": 7,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.1428445607736528,
  "event_recall": 0.2222222222222222,
  "event_precision": 0.1,
  "false_alarms_per_km": 15.7501733987471,
  "localization_mae_m": 2.0336268426978563,
  "localization_p95_m": 3.3180177538426503,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
