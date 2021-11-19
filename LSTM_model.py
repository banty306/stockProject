import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
# noinspection PyUnresolvedReferences
import silence_tensorflow.auto      # for ignoring tensorflow info and warnings
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from datetime import date

# starting and ending of data frame
start = '2010-01-01'
end = date.today().strftime('%Y-%m-%d')

# data frame
df = data.DataReader('SBI', 'yahoo', start, end)
df = df.reset_index()
df = df.drop(['Date', 'Adj Close'], axis=1)

# splitting data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

# scaling down the training data and converting it into an array
scale = MinMaxScaler(feature_range=(0, 1))
data_training_array = scale.fit_transform(data_training)

# splitting data into x_train and y_train
# x_train is taken as fist 100 values and y_train as 101 value
# then first value of x_train is dropped and y_train is inserted into x_train
# next y_train is taken as 102 value and same continues till last value
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i - 100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Simple LSTM Model
model = Sequential()

# layer 1
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

# layer 2
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

# layer 3
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

# layer 4
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

# dense layer
model.add(Dense(units=1))

# compile model with adam optimizer
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50)

# saving model
model.save('keras_model.h5')

# predicting values for testing data
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)

# scaling down the testing data and converting it into an array
input_data = scale.fit_transform(final_df)

# splitting data into x_test and y_test
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Making Prediction
y_predicted = model.predict(x_test)

# scaling up the predicted data
scale_factor = 1/scale.scale_[0]

y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# plotting original vs predicted data
plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
