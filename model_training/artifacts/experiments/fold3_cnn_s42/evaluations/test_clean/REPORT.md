# Evaluation report

Observations: 828; positives: 59; ambiguous binary labels omitted: 121.

| Metric | Value |
| --- | --- |
| average_precision | 0.3600899085491847 |
| roc_auc | 0.8591170571510436 |
| precision | 0.4186046511627907 |
| recall | 0.3050847457627119 |
| f1 | 0.35294117647058826 |
| brier | 0.06053709549774325 |
| negative_log_likelihood | 0.25659743071984153 |
| ece | 0.049571596057770874 |
| aurc_classification_error | 0.014245086234893827 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 5,
  "false_positive_events": 11,
  "false_negative_events": 4,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5835945692823914,
  "event_recall": 0.5555555555555556,
  "event_precision": 0.3125,
  "false_alarms_per_km": 6.946222356006606,
  "localization_mae_m": 3.33222184730721,
  "localization_p95_m": 4.639407316129609,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
