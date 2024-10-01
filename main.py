import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(precision=3, suppress=True)
pd.set_option('display.max_columns', None)

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

print(raw_dataset.tail())

dataset = raw_dataset.copy()
dataset = dataset.dropna()

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
print(dataset.describe().transpose())

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_label = train_features.pop('MPG')
test_label = test_features.pop('MPG')


def plot(feature1, x=None, y=None):
    plt.figure(figsize=(10, 8))
    plt.scatter(train_features[feature1], train_label, label='Data')
    if x is not None and y is not None:
        plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel(feature1)
    plt.ylabel('MPG')
    plt.legend()
    plt.show()


feature = 'Horsepower'
plot(feature)

normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
# print(normalizer.mean.numpy())

# first = np.array(train_features[:1]).astype(float)
#
# with np.printoptions(precision=2, suppress=True):
#     print('First example:', first)
#     print()
#     print('Normalized:', normalizer(first).numpy())

single_feature = np.array(train_features[feature])

single_feature_normalizer = layers.Normalization(input_shape=[1, ], axis=None)
single_feature_normalizer.adapt(single_feature)

single_feature_model = keras.Sequential([
    single_feature_normalizer,
    layers.Dense(units=1)
])

single_feature_model.summary()

single_feature_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = single_feature_model.fit(
    train_features[feature],
    train_label,
    epochs=100,
    verbose=1,
    validation_split=0.2)

single_feature_model.evaluate(
    test_features[feature],
    test_label, verbose=1)

range_min = np.min(test_features[feature]) - 10
range_max = np.max(test_features[feature]) + 10
x = tf.linspace(range_min, range_max, 200)
y = single_feature_model.predict(x)

plot(feature, x, y)

# DNN
dnn_model = keras.Sequential([
    single_feature_normalizer,
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

dnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='mean_absolute_error')

dnn_model.summary()

dnn_model.fit(
    train_features[feature], train_label,
    validation_split=0.2,
    verbose=1, epochs=100)

dnn_model.evaluate(test_features[feature], test_label, verbose=1)

x = tf.linspace(range_min, range_max, 200)
y = dnn_model.predict(x)

plot(feature, x, y)
