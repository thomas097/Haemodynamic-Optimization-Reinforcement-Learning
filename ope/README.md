Off-policy Policy Evaluation (OPE) estimators used to quantify from log-data the average performance of our models (i.e. their expected cumulative reward).

Estimators include:
- **Approximate Model Methods**: Fitted Q-Iteration (FQI) and Fitted Q-Evaluation (FQE)
- **Importance Sampling Methods**: Stepwise IS and Stepwise Weighted IS estimators (WIS)
- **Hybrid Methods**: Weighted Doubly Robust estimator (combining Weighted IS and FQI/FQE using Random Forest (RF) or Lasso Q-estimators)
