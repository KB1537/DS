import numpy as np
import matplotlib as plt
import tensorflow as tf

# Generate random training data
observations = 1000
xs = np.random.uniform(low=-10, high=10, size=(observations, 1))
zs = np.random.uniform(-10, 10, (observations, 1))

genertaed_inputs = np.column_stack((xs, zs))
noise = np.random.uniform(-1, 1, (observations, 1))

generated_targets = 2 * xs - 3*zs + 5 + noise
np.savez('TF_intro', inputs=genertaed_inputs, targets=generated_targets)

# Solve with TF
training_data = np.load('TF_intro.npz')
input_size = 2
output_size = 1
model=tf.keras.Sequential([ tf.keras.layers.Dense(output_size)])
model.compile(optimizer='SGD', loss='mean_squared_error')
model.fit(training_data['inputs'], training_data['targets'],epochs=100,verbose=2)

# Extract weights and bias 
print (model.layers[0].get_weights())

# Extract outputs(make predictions)
print(model.predict_on_batch(training_data['inputs']))

#plotting the data 
plt.pyplot(np.squeeze(model.predict_on_batch(training_data['inputs'])), np.squeeze(training_data['targets']))
plt.xlable('outputs')
plt.ylable('targets')
plt.show()