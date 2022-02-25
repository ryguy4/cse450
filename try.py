# %%
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# Keras specific
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import layers

# %%
# import stuff and preprocess
bike_first = pd.read_csv('bikes_updated.csv')
bike_first.head()
bike_first['day_of_week'] = (pd.to_datetime(bike_first['dteday']).dt.dayofweek).astype(str)
print(bike_first['day_of_week'])
bike_first.pop('dteday')
bike_first
# %%
bikes = pd.get_dummies(bike_first[['hr', 'holiday', 'workingday', 'hum', 'windspeed',
       'temp_c', 'feels_like_c', 'light_out', 'fall', 'spring',
       'summer', 'winter', 'clear', 'heavy', 'light', 'mist', 'day_of_week']])

# %%

# get target values and predictors
X = bikes
y = bike_first['total']
# split data into train and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)


# %%
# Define model

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(X))
def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(200, activation='relu'),
      layers.Dense(100, activation='relu'),
      layers.Dense(100, activation='relu'),
      layers.Dense(50, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model
train_stats = []
test_stats = []
r2_stats = []
model = build_and_compile_model(normalizer)
for _ in range(150):
    model.fit(X_train, y_train, epochs=1, use_multiprocessing=True)

    pred_train = model.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
    train_stats.append(rmse_train)
    print(rmse_train)

    pred = model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, pred))
    test_stats.append(rmse_test)
    print(rmse_test)
    r2_num = r2_score(y_test, pred)
    r2_stats.append(r2_num)
    print(r2_num)


# %%
index = [x for x in range(150)]
# %%
plt.plot(index, test_stats, color='red')
plt.plot(index, train_stats, color='blue')
plt.show()
# %%
