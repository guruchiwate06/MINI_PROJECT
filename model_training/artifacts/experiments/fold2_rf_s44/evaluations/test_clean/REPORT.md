# Evaluation report

Observations: 687; positives: 56; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.4112770400091399 |
| roc_auc | 0.8518225039619651 |
| precision | 0.22282608695652173 |
| recall | 0.7321428571428571 |
| f1 | 0.3416666666666667 |
| brier | 0.08524518791233812 |
| negative_log_likelihood | 0.2786435014878447 |
| ece | 0.11708334260369946 |
| aurc_classification_error | 0.04050190866118618 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 9,
  "false_positive_events": 30,
  "false_negative_events": 0,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5510805166310173,
  "event_recall": 1.0,
  "event_precision": 0.23076923076923078,
  "false_alarms_per_km": 19.341355705479877,
  "localization_mae_m": 4.837783413657734,
  "localization_p95_m": 8.625334016587999,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
