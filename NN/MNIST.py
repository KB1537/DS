import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Data
mnist_dataset, mnist_info = tfds.load(
    name='mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

num_validation_samples = 0.1*mnist_info.splits['train'].num_examples

# ensure interger valus is returned
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label


scaled_train_and_validation_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)

Buffer_size = 10000
shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(
    Buffer_size)

validation_data = shuffled_train_and_validation_data.take(
    num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

batch_size = 100

train_data = train_data.batch(batch_size)
validation_data = validation_data.batch(num_validation_samples)

validation_inputs, validation_targets = next(iter(validation_data))

# model
input_size = 784
output_size = 10
hidden_layer_size = 50

model = tf.keras.Sequential([
    tf.keras.layer.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax'),
])

# choose optimizer and loss function
model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
Num_epochs = 5
model.fit(train_data, epochs=Num_epochs, validation_data=(
    validation_inputs, validation_targets), verbose=2)

# Testing
Test_loss, test_accuracy = model.evaluate(test_data)
print('Test_loss:{0:.2f}. Test accuracy: {1:.2f}%'.format(Test_loss, test_accuracy*100.))

