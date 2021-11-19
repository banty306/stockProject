import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
# noinspection PyUnresolvedReferences
import silence_tensorflow.auto      # for ignoring tensorflow info and warnings
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from datetime import date

# starting and ending of data frame
start = '2010-01-01'
end = date.today().strftime('%Y-%m-%d')

# decoration
st.title('Stock Trend Prediction')

# data frame
user_input = st.text_input('Enter Stock Ticker', 'SBI')
df = data.DataReader(user_input, 'yahoo', start, end)

print(df)

# Describing Data
st.subheader('Data from '+start.split('-')[0]+' - '+end.split('-')[0])
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

# splitting data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

# scaling down the training data and converting it into an array
scale = MinMaxScaler(feature_range=(0, 1))
data_training_array = scale.fit_transform(data_training)

# Load the model
model = load_model('keras_model.h5')

# testing data
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

# Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
