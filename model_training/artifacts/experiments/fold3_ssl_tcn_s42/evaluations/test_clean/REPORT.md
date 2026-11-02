# Evaluation report

Observations: 828; positives: 59; ambiguous binary labels omitted: 121.

| Metric | Value |
| --- | --- |
| average_precision | 0.39880949803755833 |
| roc_auc | 0.795970994688237 |
| precision | 0.44642857142857145 |
| recall | 0.423728813559322 |
| f1 | 0.43478260869565216 |
| brier | 0.10924024323458512 |
| negative_log_likelihood | 0.3752613948796119 |
| ece | 0.2015910253521837 |
| aurc_classification_error | 0.06110850042870223 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 5,
  "false_positive_events": 16,
  "false_negative_events": 4,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5835945692823914,
  "event_recall": 0.5555555555555556,
  "event_precision": 0.23809523809523808,
  "false_alarms_per_km": 10.103596154191427,
  "localization_mae_m": 3.503797908274646,
  "localization_p95_m": 5.692465631831691,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
