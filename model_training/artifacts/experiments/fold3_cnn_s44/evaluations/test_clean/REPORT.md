# Evaluation report

Observations: 828; positives: 59; ambiguous binary labels omitted: 121.

| Metric | Value |
| --- | --- |
| average_precision | 0.19914444473774492 |
| roc_auc | 0.7719027572678583 |
| precision | 0.0 |
| recall | 0.0 |
| f1 | 0.0 |
| brier | 0.06483201655958262 |
| negative_log_likelihood | 0.2548340511530892 |
| ece | 0.04014168544644568 |
| aurc_classification_error | 0.025842696809502227 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 1,
  "false_positive_events": 2,
  "false_negative_events": 8,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5835945692823914,
  "event_recall": 0.1111111111111111,
  "event_precision": 0.3333333333333333,
  "false_alarms_per_km": 1.2629495192739284,
  "localization_mae_m": 6.1159292825761895,
  "localization_p95_m": 6.1159292825761895,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
