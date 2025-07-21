# volatility-and-direction-prediction-using-online-learning
This project explores the use of online machine learning models to predict both market volatility (as a regression task) and market direction (as a binary classification task) from financial time series data.

Unlike traditional batch learning, which trains on large datasets all at once, online learning updates the model incrementally as new data arrives. This makes it ideal for real-time systems, streaming data, and environments where the data distribution may shift over time. The model does not need to be retrained from scratch and can adapt quickly to changing patterns in financial markets.

Key Features
Dual-task prediction: Simultaneously models volatility (continuous target) and market direction (up or down).

Online learning architecture: Uses data streams to update models one sample at a time.

Real-time adaptivity: Models are designed to handle drift and evolving distributions.

Evaluation on both full dataset and rolling window to assess long-term and short-term performance.

Rich set of metrics: Includes accuracy, precision, recall, F1, ROC AUC, and geometric mean for classification, and MAE, RMSE, and R² for regression.

Why Online Learning?
Financial markets are dynamic. Patterns and regimes change rapidly. Models trained offline on static historical data often degrade in performance when exposed to new, unseen data. Online learning provides a solution by:

Continuously adapting to concept drift and volatility shifts

Requiring low memory and computational resources

Enabling low-latency predictions for real-time systems

Providing flexibility in environments with delayed or partial feedback

Core Techniques and Tools
Data Source: Financial time series from yfinance, including prices and derived indicators.

Online Models:

River library for incremental learning algorithms

ARFClassifier for direction prediction

LinearRegression with online optimizers for volatility estimation

Drift Detection:

Integrated drift detectors for adjusting to distributional shifts

Metric Tracking:

Real-time tracking of performance using river.metrics

Separate tracking for long-term and rolling window evaluation

sample  results
Confusion Matrix (Direction Prediction):
[[ 21 611]
 [ 22 752]]

Final Metrics:
vol_mae: 0.0009
vol_rmse: 0.0018
vol_r2: 0.9375
dir_accuracy: 0.5498
dir_precision: 0.5517
dir_recall: 0.9716
dir_f1: 0.7038
dir_roc_auc: 0.5019
dir_geometric_mean: 0.1797

Rolling Metrics:
vol_mae: 0.0005
dir_accuracy: 0.6000
dir_f1: 0.7500
