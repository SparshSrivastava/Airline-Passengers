import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading the data
dataset = pd.read_csv('international-airline-passengers.csv')
dataset.columns = ['Month', 'Passengers']
dataset.drop('Month', inplace = True, axis = 1)

dataset = np.array(dataset).reshape(-1, 1)

# Scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset)

# Splitting the dataset into training and testing
print(dataset.shape)

# Plotting our data
plt.plot(dataset)
plt.savefig('Passengers')

# Defining the size for the training and testing
train_size = int(len(dataset) * 0.67)
remaining = len(dataset) - train_size
validation_size = int(remaining * 0.67)
test_size = remaining - validation_size

train = dataset[:train_size, 0]
test = dataset[train_size:, 0]
validation = test[:validation_size]
test = test[validation_size:]

# Defining a function to create our dataset
def create_dataset(dataset, look_up):
    data_X, data_Y = [], []
    for i in range(len(dataset) - look_up - 1):
        x = dataset[i:(i+look_up)]
        data_X.append(x)
        y = dataset[i+look_up]
        data_Y.append(y)
    return np.array(data_X), np.array(data_Y)

look_up = 1
train_X, train_Y = create_dataset(train, look_up)
test_X, test_Y = create_dataset(test, look_up)
validation_X, validation_Y = create_dataset(validation, look_up)

# Building our model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(20, activation='tanh', input_shape = (look_up, 1), return_sequences= True))
model.add(Dropout(0.1))
model.add(LSTM(10, activation = 'tanh'))

model.add(Dense(20))
model.add(Dense(1)) 

print(model.summary())

train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], 1))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], 1))
validation_X = validation_X.reshape(((validation_X.shape[0], validation_X.shape[1], 1)))

train_Y = train_Y.reshape(-1, 1)
test_Y = test_Y.reshape(-1, 1)
validation_Y = validation_Y.reshape(-1, 1)

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', min_delta=3e-7)

model.compile(optimizer='adam', loss = 'mean_squared_error', metrics = ['accuracy'])
model.fit(train_X, train_Y, epochs = 80, batch_size=look_up, validation_data=(validation_X, validation_Y))
# Making the predictions 
y_pred = model.predict(test_X)

y_pred = scaler.inverse_transform(y_pred)

test_Y = np.array(test_Y)
test_Y = scaler.inverse_transform(test_Y)

# Plotting the predictions along with the actual values
plt.figure(figsize=(14, 5))
plt.plot(y_pred, label = 'Predicted values of passengers')
plt.plot(test_Y, label = 'Actual values of Y')
plt.ylabel(' Number of Passengers')
plt.legend()
plt.savefig('Predicted_3.png')
