import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Dropout
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Dense, Flatten, Dropout

#folder_path = "D:/2nd objective data/x_y_arrays"

X_data_array = np.load('X_data_array.npy')
Y_data_array = np.load('Y_data_array.npy')

# Reshaping X_data_array
X_data_reshaped = np.transpose(X_data_array, (1, 2, 3, 0))
X_data_reshaped = np.expand_dims(X_data_reshaped, axis=0)  # Adding the samples dimension

# Reshaping Y_data_array
Y_data_reshaped = np.expand_dims(Y_data_array, axis=(0, -1))

Y_data_reshaped.shape, X_data_reshaped.shape



Y_data_reshaped[0][0][20][20]
Y_data_reshaped[0][0][20][0]


# Removing the leading dimension of 1 and reshaping the data (was redundant)
X_data_reshaped_new = X_data_reshaped.squeeze(0)
Y_data_reshaped_new = Y_data_reshaped.squeeze(0)

X_data_reshaped = X_data_reshaped_new
Y_data_reshaped = Y_data_reshaped_new


X_data_reshaped.shape, Y_data_reshaped.shape

#Need to take care of nan values

X_data_reshaped[np.isnan(X_data_reshaped)] = 0
Y_data_reshaped[np.isnan(Y_data_reshaped)] = 0


###Converting to 3 day forecasting tasks


def reshape_data_sliding_window(X_data, Y_data, window_size, pred_size=3):
    X_sequences = []
    Y_sequences = []
    for i in range(X_data.shape[0] - window_size - pred_size + 1):
        X_sequences.append(X_data[i:i+window_size])
        Y_sequences.append(Y_data[i+window_size:i+window_size+pred_size])
    return np.array(X_sequences), np.array(Y_sequences)

WINDOW_SIZE = 10
X_data_sequences, Y_data_next_3_days = reshape_data_sliding_window(X_data_reshaped, Y_data_reshaped, WINDOW_SIZE)


# Define a learning rate schedule
learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.00001,
    decay_steps=10000,
    decay_rate=0.96
)

# Create an optimizer with the learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

#Baseline 2 - Fully Connected Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape

def fully_connected_model():
    model = Sequential([
        Flatten(input_shape=(10, 45, 36, 7)),
        Dense(128, activation='relu'),
        Dense(45*36*3),  # Output layer for 3 days, each with a 45x36 image flattened
        Reshape((3, 45, 36, 1))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


model = fully_connected_model()
model.summary()
#history=model.fit(X_data_sequences, Y_data_next_3_days, epochs=40, batch_size=10, validation_split=0.2)
model.fit(X_data_sequences, Y_data_next_3_days, epochs=40, batch_size=10, validation_split=0.2)


# Save the model in the HDF5 format
model.save("regression_3_B2_Cauveri.h5")  # It will save the model in a single HDF5 file