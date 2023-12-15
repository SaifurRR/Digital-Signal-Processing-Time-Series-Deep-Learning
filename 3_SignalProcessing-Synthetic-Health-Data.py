import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Generate synthetic health data
np.random.seed(42)
fs = 1000  # Sampling frequency
t = np.arange(0, 10, 1/fs)  # Time vector
f1, f2 = 5, 20  # Frequencies of two signals
signal1 = np.sin(2 * np.pi * f1 * t)  # Signal 1
signal2 = np.sin(2 * np.pi * f2 * t)  # Signal 2
noisy_signal = signal1 + 0.5 * signal2 + 0.2 * np.random.randn(len(t))  # Noisy composite signal

# Apply a bandpass filter to extract the relevant frequency band
nyq = 0.5 * fs
low = 3
high = 8
b, a = signal.butter(4, [low / nyq, high / nyq], btype='band')
filtered_signal = signal.filtfilt(b, a, noisy_signal)

# Extract features (e.g., statistical measures)
mean_value = np.mean(filtered_signal)
std_dev = np.std(filtered_signal)
max_value = np.max(filtered_signal)
min_value = np.min(filtered_signal)

# Combine features into a feature vector
feature_vector = [mean_value, std_dev, max_value, min_value]

# Create labels (0: healthy, 1: unhealthy)
labels = np.random.randint(2, size=len(t))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(np.array([feature_vector] * len(t)),
                                                    labels,
                                                    test_size=0.2,
                                                    random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a machine learning model (Random Forest in this example)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

# Print results
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)

# Plot the signals
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, noisy_signal, label='Noisy Signal')
plt.plot(t, filtered_signal, label='Filtered Signal')
plt.title('Original and Filtered Signals')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, labels, label='Labels', marker='o', linestyle='None')
plt.title('Health Labels (0: Healthy, 1: Unhealthy)')
plt.legend()

plt.tight_layout()
plt.show()
