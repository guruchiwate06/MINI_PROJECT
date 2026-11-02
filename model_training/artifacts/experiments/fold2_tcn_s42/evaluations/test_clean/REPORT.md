# Evaluation report

Observations: 687; positives: 56; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.24152966088435773 |
| roc_auc | 0.7252660176590446 |
| precision | 0.19424460431654678 |
| recall | 0.48214285714285715 |
| f1 | 0.27692307692307694 |
| brier | 0.12622005293412775 |
| negative_log_likelihood | 0.4090984804137498 |
| ece | 0.15625567873262006 |
| aurc_classification_error | 0.08106646000826692 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 4,
  "false_positive_events": 33,
  "false_negative_events": 5,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5510805166310173,
  "event_recall": 0.4444444444444444,
  "event_precision": 0.10810810810810811,
  "false_alarms_per_km": 21.275491276027864,
  "localization_mae_m": 4.941327903412173,
  "localization_p95_m": 9.322657996305281,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
