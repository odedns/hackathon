# Time series anomaly detection
This tutorial is an introduction to time series anomaly detection using TensorFlow.

Anomaly detection is a problem with implementations in a wide variety of domains. It involves the identification of
novel or unexpected observations or sequences within the data being captured. 

In this tutorial different styles of LSTM models are trained using a series of data stock market prices.
Reconstruction error between the original and predicted data is used to determine the anomalies.
In the last step, anomalies are plotted over the original data to visualize the results.
### Covid19
affected the markets so was not a surprise that the algorithms found most novel market
behavior over the outbreaks waves and lockdowns.

This tutorial includes five sections:

1. Data preprocessing.
2. Training on LSTM neural network. LSTM stands for Long-Short Term Memory
3. Reconstruction of the data based on LSTM prediction functionality.
4. Mean Absolute Error (MAE) computation between original and reconstructed data.
5. Anomalies detection and plotting.
