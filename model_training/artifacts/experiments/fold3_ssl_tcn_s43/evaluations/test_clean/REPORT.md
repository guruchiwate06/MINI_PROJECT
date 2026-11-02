# Evaluation report

Observations: 828; positives: 59; ambiguous binary labels omitted: 121.

| Metric | Value |
| --- | --- |
| average_precision | 0.41902184399215764 |
| roc_auc | 0.8429834034956251 |
| precision | 0.5853658536585366 |
| recall | 0.4067796610169492 |
| f1 | 0.48 |
| brier | 0.22282281276111954 |
| negative_log_likelihood | 0.620444639828819 |
| ece | 0.3669532026974679 |
| aurc_classification_error | 0.06599378536177974 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 5,
  "false_positive_events": 6,
  "false_negative_events": 4,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5835945692823914,
  "event_recall": 0.5555555555555556,
  "event_precision": 0.45454545454545453,
  "false_alarms_per_km": 3.7888485578217854,
  "localization_mae_m": 0.8283992554093402,
  "localization_p95_m": 2.201552156125035,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
