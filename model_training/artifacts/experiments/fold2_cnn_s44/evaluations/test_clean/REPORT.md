# Evaluation report

Observations: 687; positives: 56; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.22867998475738827 |
| roc_auc | 0.821371971926647 |
| precision | 0.2608695652173913 |
| recall | 0.32142857142857145 |
| f1 | 0.288 |
| brier | 0.2604795678256344 |
| negative_log_likelihood | 0.7158010437014638 |
| ece | 0.39483831261143515 |
| aurc_classification_error | 0.1573577797493391 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 6,
  "false_positive_events": 27,
  "false_negative_events": 3,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5510805166310173,
  "event_recall": 0.6666666666666666,
  "event_precision": 0.18181818181818182,
  "false_alarms_per_km": 17.407220134931887,
  "localization_mae_m": 5.067549272046629,
  "localization_p95_m": 7.393369351450104,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
