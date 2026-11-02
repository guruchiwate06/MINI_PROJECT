# Evaluation report

Observations: 828; positives: 59; ambiguous binary labels omitted: 121.

| Metric | Value |
| --- | --- |
| average_precision | 0.284572817304909 |
| roc_auc | 0.8094377465782108 |
| precision | 0.3333333333333333 |
| recall | 0.2542372881355932 |
| f1 | 0.28846153846153844 |
| brier | 0.09065588656569891 |
| negative_log_likelihood | 0.30239788467031076 |
| ece | 0.0845359296129174 |
| aurc_classification_error | 0.02680801400052197 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 5,
  "false_positive_events": 22,
  "false_negative_events": 4,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5835945692823914,
  "event_recall": 0.5555555555555556,
  "event_precision": 0.18518518518518517,
  "false_alarms_per_km": 13.892444712013212,
  "localization_mae_m": 2.382016222909897,
  "localization_p95_m": 5.45731316486217,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
