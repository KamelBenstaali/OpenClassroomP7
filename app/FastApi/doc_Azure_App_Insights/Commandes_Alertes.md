## Requête pour voir les tweets faussement prédits

```
traces
| where message == "prediction_feedback_reported"
| project timestamp,
          text            = tostri(customDimensions.text),
          predicted_label = tostring(customDimensions.predicted_label),
          expected_label  = tostring(customDimensions.expected_label),
          score           = todouble(customDimensions.score),
          comment         = tostring(customDimensions.comment),
          is_correct      = tobool(customDimensions.is_correct)
| where tobool(is_correct) == false
| sort by timestamp desc;
```

## Requête pour lancer l'alerte lorsqu'on trace 3 tweets mal prédits ou plus
```
traces
| where message == "prediction_feedback_reported"
| where tobool(customDimensions.is_correct) == false
| summarize cnt = count() by bin(timestamp, 5m)
| where cnt >= 3;
```

## Requête pour lancer l'alerte lorsqu'on trace 3 tweets mal prédits ou plus avec les informations de ces tweets
```
let feedback =
    traces
    | where message == "prediction_feedback_reported"
    | extend is_correct = tobool(customDimensions.is_correct)
    | where is_correct == false
    | extend
        window          = bin(timestamp, 5m),
        text            = tostring(customDimensions.text),
        predicted_label = tostring(customDimensions.predicted_label),
        expected_label  = tostring(customDimensions.expected_label),
        score           = todouble(customDimensions.score),
        comment         = tostring(customDimensions.comment)
    | project timestamp, window, text, predicted_label, expected_label, score, comment;
feedback
| summarize cnt = count() by window
| where cnt >= 3                      // seuil d’alerte
| join kind=inner feedback on window
| project window, timestamp, text, predicted_label, expected_label, score, comment
| order by window desc, timestamp desc
```