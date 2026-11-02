# Evaluation report

Observations: 828; positives: 59; ambiguous binary labels omitted: 121.

| Metric | Value |
| --- | --- |
| average_precision | 0.4189638051129162 |
| roc_auc | 0.823742037865597 |
| precision | 0.5357142857142857 |
| recall | 0.2542372881355932 |
| f1 | 0.3448275862068966 |
| brier | 0.20346922850153845 |
| negative_log_likelihood | 0.5570350800098444 |
| ece | 0.30871139286867827 |
| aurc_classification_error | 0.05568063541305007 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 5,
  "false_positive_events": 12,
  "false_negative_events": 4,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5835945692823914,
  "event_recall": 0.5555555555555556,
  "event_precision": 0.29411764705882354,
  "false_alarms_per_km": 7.577697115643571,
  "localization_mae_m": 1.4991677645698929,
  "localization_p95_m": 3.1630900240365007,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
