# Arm A: pure QELM (L = 1) on engineered + raw features
- the classical engineered features (Linh's rolling features) carry the memory. QELM acts as an untrained quantum feature map with a trained classical linear readout.
- should be able to compete with random features, which are obtained by replacing the quantum feature map with a randomly drawn nonlinear map (tanh or Fourier). Importantly both maps should produce the same number of features and be evaluated with the same readout, pairing results by seed.

# Arm B: reset-window QRC (L > 1) on raw features only, with and without input-qubit reset
- from the case description, this was the coaches' hope: the quantum reservoir helps with the "temporal memory".
- if we only input the transactions raw amounts and scenario 3 fraud is detected, we have a relevant result.
- it's unlikely that it competes with the classical baseline over all scenarios



# Objectives

## Main

The Average Precision (AP) score is our main objective. We should tune our model by evaluating it on the validation set (averaging over N seeds).

## Ultimate goal

If we want to do any quantum-vs-classical comparison, we shall refer to the difference in AP score between quantum and classical (e.g. random features with equal width, XGBoost with same tuning budget, etc.).

## Scores by fraud scenarios

By discriminating scores (in particular AP) by fraud scenarios (i.e. 1, 2, or 3), we will have a better picture of where our model wins or loses. For example, a good AP for scenario 3 in Arm B will indicate the QRC has memory.

## Operational

We can also report the [Card Precision Top-k](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_4_PerformanceMetrics/TopKBased.html#card-precision-top-k), which is a more operational performance metric.

## Cost

- total number of qubits (input + memory)
- circuit depth
- observables
- number of shots
- wall time

## Other

F1, MCC, precision, recall (threshold chosen on validation), ROC-AUC
