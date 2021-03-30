from tensorflow.keras import Model, Input, losses, optimizers, metrics, utils
from tensorflow.keras.layers import Flatten, Dense, Dropout, Concatenate, Conv1D, MaxPool1D
from tensorflow import float32, string
from tensorflow.keras.layers.experimental.preprocessing import Normalization, StringLookup, CategoryEncoding

import pandas as pd
import numpy as np

data = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")

from sklearn.model_selection import train_test_split

labels = data.pop('survived')
label_names = ["Not survived", "Survived"]

features = {}

# Converting CSV file into Tensorflow object

for name, column in data.items():
  dtype = column.dtype
  if dtype == object:
    dtype = string
  else:
    dtype = float32
  features[name] = Input(shape = (1,), name=name, dtype=dtype)

# Extracting and normalizing numeric features
numeric_features = {name:feature for name, feature in features.items() if feature.dtype == float32}

x = Concatenate()(list(numeric_features.values()))
norm = Normalization()
norm.adapt(np.array(data[numeric_features.keys()]))
numeric_features = norm(x)

processed_features = [numeric_features]

# Extracting and normalizing non-numeric features

for name, feature in features.items():
  if feature.dtype == float32:
    continue
  word = StringLookup(vocabulary=np.unique(data[name]))
  one_hot = CategoryEncoding(max_tokens=word.vocab_size())

  x = word(feature)
  x = one_hot(x)
  processed_features.append(x)

processed_features = Concatenate()(processed_features)
processed_features = Model(features, processed_features)


utils.plot_model(model=processed_features, rankdir= 'LR', dpi=72, show_shapes=True)

feature_dict = {name:np.array(value) for name, value in data.items()}

train_features, test_features, train_labels, test_labels = train_test_split(processed_features(feature_dict).numpy(), labels, test_size = 0.2)

class Mymodel(Model):
  def __init__(self, in_shape = (20,), out = 1):
    super(Mymodel, self).__init__()
    self.inp = Flatten()
    self.out = Dense(out, activation='sigmoid')
    self.h1 = Dense(32, activation='relu')
    self.h2 = Dense(64, activation = 'relu')
    self.h3 = Dense(128, activation='relu')
  
  def __call__(self, x):
    x = self.inp(x)
    x = self.h1(x)
    x = self.h2(x)
    #x = self.h3(x)
    return self.out(x)
model = Mymodel((None,33), 1)

lr = 0.001
beta_1 = 0.8
beta_2 = 0.9

loss_fn = losses.BinaryCrossentropy()
optimizer = optimizers.Adam(learning_rate = lr)

train_loss = metrics.Mean(name = 'train_loss')
train_accuracy = metrics.BinaryAccuracy(name = 'train_accuracy')

test_loss = metrics.Mean(name = 'test_loss')
test_accuracy = metrics.BinaryAccuracy(name = 'test_accuracy')

from tensorflow.keras import Sequential, callbacks, regularizers
BATCH_SIZE = 8
simple_model = Sequential([
                           Input(shape = (33,)),
                           Dense(128, activation='relu'),
                           Dropout(0.5),
                           Dense(64, activation='sigmoid'),
                           Dense(32, activation='relu'),
                           Dense(1, activation='sigmoid')
])
stop = callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
simple_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
hist_simple = simple_model.fit(train_features, train_labels, epochs=500, callbacks=stop, validation_split=0.15, batch_size=BATCH_SIZE)

predictions = simple_model.predict(test_features)
for i in range(10):
  print(f"Prediction: {label_names[int(np.round(predictions[i][0]))]}")

test_loss, test_acc = simple_model.evaluate(test_features, test_labels)

simple_model.save('simple_model')

from tensorflow import function, GradientTape
@function
def train(features, labels):
  with GradientTape() as tape:
    predictions = model(features)
    loss = loss_fn(labels, predictions)
  grad = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grad, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@function
def test(features, labels):
  predictions = model(features)
  loss = loss_fn(labels, predictions)

  test_loss(loss)
  test_accuracy(labels, predictions)

EPOCHS = 100
from tensorflow.data import Dataset
train_ds = Dataset.from_tensor_slices(
    (train_features, train_labels)).shuffle(10000).batch(32)

test_ds = Dataset.from_tensor_slices(
    (test_features, test_labels)).shuffle(10000).batch(32)

train_loss_vec = []
test_loss_vec = []
train_acc_vec = []
test_acc_vec = []

for epoch in range(EPOCHS):
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()
  for features, labels in train_ds:
    train(features, labels)
  for features, labels in test_ds:
    test(features, labels)
  
  train_loss_vec.append(train_loss)
  test_loss_vec.append(test_loss)

  train_acc_vec.append(train_accuracy)
  test_acc_vec.append(test_accuracy)

  if epoch % 10 == 0:
    print(
      f'Epoch {epoch + 1}.2f, '
      f'Loss: {train_loss.result():.2f}, '
      f'Accuracy: {train_accuracy.result() * 100:.2f}, '
      f'Test Loss: {test_loss.result():.2f}, '
      f'Test Accuracy: {test_accuracy.result() * 100:.2f}'
    )

