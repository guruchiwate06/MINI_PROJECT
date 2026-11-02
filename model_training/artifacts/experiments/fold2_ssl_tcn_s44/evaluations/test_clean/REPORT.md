# Evaluation report

Observations: 687; positives: 56; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.22138654475660843 |
| roc_auc | 0.7192098709531356 |
| precision | 0.25 |
| recall | 0.19642857142857142 |
| f1 | 0.22 |
| brier | 0.1729636801014708 |
| negative_log_likelihood | 0.5292930049334775 |
| ece | 0.2654845037047097 |
| aurc_classification_error | 0.11592208978708943 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 2,
  "false_positive_events": 14,
  "false_negative_events": 7,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5510805166310173,
  "event_recall": 0.2222222222222222,
  "event_precision": 0.125,
  "false_alarms_per_km": 9.025965995890608,
  "localization_mae_m": 6.837279506235774,
  "localization_p95_m": 7.84627133187208,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
