## 2025.08.26

1. Experimented with training of QwenOmni gate head only on ~800 audios samples 3 sec each * 5 chunks or roughly ~4000 training samples.

Model scored badly:
```
threshold = 0.5 

--- Evaluation Complete ---
Overall Accuracy: 54.83% (2628 / 4793)

Confusion Matrix (Positive = TRANSLATE):
  TP:  428   TN: 2200   FP: 1722   FN:  443

Metrics (TRANSLATE):
  Precision: 0.1991
  Recall:    0.4914
  F1-Score:  0.2833

Total runtime: 616.64 s
```
Basically that means:


## 2025.08.27

State: Experiments without fundumentals is uselss, so do more research first idiota.
---
common_voice_17_train set:
INFO:__main__:Total clips processed: 30845
INFO:__main__:Total chunks generated: 175753
INFO:__main__:WAIT chunks: 144908 (82.4%)
INFO:__main__:TRANSLATE chunks: 30845 (17.6%)
INFO:__main__:Average chunks per clip: 5.70
---

1. Experimented with fully training with common_voice_17_train set. 
Eval results:
1 epoch
```
threshold = 0.5
--- Evaluation Complete ---
Overall Accuracy: 69.77% (3344 / 4793)

Confusion Matrix (Positive = TRANSLATE):
  TP:  387   TN: 2957   FP:  965   FN:  484

Metrics (TRANSLATE):
  Precision: 0.2862
  Recall:    0.4443
  F1-Score:  0.3482

Total runtime: 630.06 s
```

## 2025.08.28

1. Experimented with fully training with common_voice_17_train set. 
Eval results:
3 epochs
```
threshold = 0.5
--- Evaluation Complete ---
Overall Accuracy: 63.59% (3048 / 4793)

Confusion Matrix (Positive = TRANSLATE):
  TP:  516   TN: 2532   FP: 1390   FN:  355

Metrics (TRANSLATE):
  Precision: 0.2707
  Recall:    0.5924
  F1-Score:  0.3716

Total runtime: 640.25 s
```