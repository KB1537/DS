import numpy as np
import tensorflow as tf
from sklearn import preprocessing

# Extract data from csv
df = np.loadtxt('Audiobooks_data.csv', delimiter=',')
unscaled_inputs_all = df[:, 1:-1]
targets_all = df[:, -1]

# balance dataset
num_one_targets = int(np.sum(targets_all))
zero_targets_counter = 0
indices_to_remove = []

for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)

# standerise inputs
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

# shuffle data
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]


# split data into train,validate,test
samples_count = shuffled_inputs.shape[0]

train_sample_count = int(0.8*samples_count)
validation_sample_count = int(0.1*samples_count)
test_sample_count = samples_count - train_sample_count - validation_sample_count

train_inputs = shuffled_inputs[:train_sample_count]
train_targets = shuffled_targets[:train_sample_count]

validation_inputs = shuffled_inputs[train_sample_count:train_sample_count+validation_sample_count]
validation_targets = shuffled_targets[train_sample_count:train_sample_count+validation_sample_count]

test_inputs = shuffled_inputs[train_sample_count+validation_sample_count:]
test_targets = shuffled_targets[train_sample_count+validation_sample_count:]

print(np.sum(train_targets), train_sample_count, np.sum(train_targets) / train_sample_count)

print(np.sum(validation_targets), validation_sample_count,np.sum(validation_targets) / validation_sample_count)

print(np.sum(test_targets), test_sample_count,np.sum(test_targets) / test_sample_count)

# save datasets as npz
np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation',inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)

npz= np.load('Audiobooks_data_train.npz')

train_input=npz['inputs'].astype(np.float32)
train_target=npz['targets'].astype(np.int64)

npz=np.load('Audiobooks_data_validation.npz')

validation_inputs,validation_targets=npz['inputs'].astype(np.float32), npz['targets'].astype(np.int64)

npz=np.load('Audiobooks_data_test.npz')

test_input,test_target=npz['inputs'].astype(np.float32),npz['targets'].astype(np.int64)


#Model
input_size = 10
output_size = 2
hidden_layer_size = 50

model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax'),
])
# choose optimizer and loss function
model.compile(optimizer="adam",loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size=100
max_epochs=100
early_stopping= tf.keras.callbacks.EarlyStopping(patience=2)

model.fit(train_inputs,
          train_targets,
          batch_size=batch_size,
          epochs=max_epochs,
          callbacks=[early_stopping],
          validation_data=(validation_inputs,validation_targets),
          verbose=2)

#Test model
test_loss, test_accuracy=model.evaluate(test_inputs,test_targets)