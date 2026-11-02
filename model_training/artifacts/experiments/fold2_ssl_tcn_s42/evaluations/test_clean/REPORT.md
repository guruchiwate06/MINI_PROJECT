# Evaluation report

Observations: 687; positives: 56; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.1821874566339629 |
| roc_auc | 0.5842483586144442 |
| precision | 0.2631578947368421 |
| recall | 0.17857142857142858 |
| f1 | 0.2127659574468085 |
| brier | 0.11706987546496775 |
| negative_log_likelihood | 0.3967098710239969 |
| ece | 0.1063535179332468 |
| aurc_classification_error | 0.06624855680684807 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 3,
  "false_positive_events": 17,
  "false_negative_events": 6,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5510805166310173,
  "event_recall": 0.3333333333333333,
  "event_precision": 0.15,
  "false_alarms_per_km": 10.960101566438595,
  "localization_mae_m": 4.102733123868727,
  "localization_p95_m": 5.60946784045002,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
