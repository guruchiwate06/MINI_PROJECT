# Evaluation report

Observations: 687; positives: 56; ambiguous binary labels omitted: 113.

| Metric | Value |
| --- | --- |
| average_precision | 0.2758752393010893 |
| roc_auc | 0.7515281865519584 |
| precision | 0.3170731707317073 |
| recall | 0.23214285714285715 |
| f1 | 0.26804123711340205 |
| brier | 0.08847239981526901 |
| negative_log_likelihood | 0.3015012426961509 |
| ece | 0.06470870046760661 |
| aurc_classification_error | 0.036811700280260505 |

Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.

```json
{
  "true_positive_events": 4,
  "false_positive_events": 14,
  "false_negative_events": 5,
  "covered_site_traversals": 9,
  "evaluated_distance_km": 1.5510805166310173,
  "event_recall": 0.4444444444444444,
  "event_precision": 0.2222222222222222,
  "false_alarms_per_km": 9.025965995890608,
  "localization_mae_m": 4.081275758058069,
  "localization_p95_m": 6.524684247423551,
  "target_semantics": "nominal documented site proximity, not independently adjudicated frame defects"
}
```

Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.
