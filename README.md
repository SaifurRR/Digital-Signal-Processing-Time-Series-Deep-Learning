# Time-Series-Deep-Learning

### 1. Time Sequence Classification Using Deep Learning
This project shows how to classify sequence data using a long short-term memory (LSTM) network. To train a deep neural network to classify sequence data, we use an LSTM network. An LSTM network enables you to input sequence data into a network and make predictions based on the individual time steps of the sequence data.
This project uses the Waveform data set to train the LSTM network to recognize the type of waveform given Time Series data. The training data contains Time Series data for four types of waveform. Each sequence has three channels and varies in length.

### 2. Classify ECG Signals Using Long Short-Term Memory Networks with GPU Acceleration
This project shows how to build a classifier to detect Atrial Fibrillation in ECG signals using an LSTM network.
The procedure uses oversampling to avoid the classification bias that occurs when one tries to detect abnormal
conditions in populations composed mainly of healthy patients. Training the LSTM network using raw signal
data results in poor classification accuracy. Training the network using two Time-Frequency-moment features for
each signal significantly improves the Classification performance and also decreases the training time.

### 3. Signal-Processing-Synthetic-Health-Data

Classification Report:
               precision    recall  f1-score   support

           0       0.00      0.00      0.00      1005
           1       0.50      1.00      0.66       995

    accuracy                           0.50      2000
   macro avg       0.25      0.50      0.33      2000
weighted avg       0.25      0.50      0.33      2000
